"""ffmpeg/ffprobe plumbing shared by every tool: probing, running ffmpeg with
progress and cancellation, grabbing single frames, and decoding tiny grayscale
frame sequences for hashing. No web, no database.
"""
from __future__ import annotations

import functools
import io
import json
import re
import subprocess
import threading

import numpy as np
from PIL import Image

from .jobs import Cancelled

# Hide console windows spawned by ffmpeg when running under pythonw on Windows.
_POPEN_KW = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


@functools.lru_cache(maxsize=1)
def ffmpeg_version() -> tuple[int, int]:
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, **_POPEN_KW).stdout
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH")
    m = re.search(r"version n?(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (99, 0)  # git builds: assume new


def vfr_args() -> list[str]:
    """Variable-frame-rate output flag (-fps_mode arrived in ffmpeg 5.1)."""
    return ["-fps_mode", "vfr"] if ffmpeg_version() >= (5, 1) else ["-vsync", "vfr"]


def _frac(s: str | None) -> float:
    num, _, den = (s or "0/0").partition("/")
    try:
        num, den = float(num or 0), float(den or 1)
    except ValueError:
        return 0.0
    return num / den if den and num else 0.0


def probe(path: str) -> dict:
    """Geometry, rate, duration, bitrate, codecs and audio streams of a video file."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,"
         "channels,sample_rate,duration:stream_disposition=attached_pic",
         "-show_entries", "format=duration,bit_rate,format_name", "-of", "json", path],
        capture_output=True, text=True, **_POPEN_KW)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "ffprobe failed").strip()[-300:])
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"
            and not (s.get("disposition") or {}).get("attached_pic")]
    if not vids:
        raise RuntimeError("no video stream")
    vid = vids[0]
    fps_frac = vid.get("avg_frame_rate")
    if not _frac(fps_frac):  # some containers leave avg empty
        fps_frac = vid.get("r_frame_rate")
    fps = _frac(fps_frac)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or vid.get("duration") or 0)
    audio = [{"index": s["index"], "codec": s.get("codec_name"),
              "channels": s.get("channels"), "sample_rate": s.get("sample_rate")}
             for s in streams if s.get("codec_type") == "audio"]
    return {
        "width": vid.get("width") or 0, "height": vid.get("height") or 0,
        "fps_frac": fps_frac if fps else None, "fps": round(fps, 3),
        "duration": duration,
        "bitrate": int(fmt.get("bit_rate") or 0),
        "vcodec": vid.get("codec_name"),
        "container": fmt.get("format_name"),
        "has_audio": bool(audio), "audio": audio,
    }


def run(cmd: list[str], job=None, duration: float = 0, on_stderr=None) -> None:
    """Run ffmpeg with `-progress pipe:1` injected, reporting progress in seconds.

    stderr is drained on a thread (so showinfo-style output can't fill the pipe
    and stall ffmpeg); each line goes to on_stderr and the tail is kept for
    the error message.
    """
    cmd = [cmd[0], "-progress", "pipe:1", "-nostats"] + cmd[1:]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", **_POPEN_KW)
    tail: list[str] = []

    def drain():
        for line in proc.stderr:
            if on_stderr:
                on_stderr(line)
            tail.append(line)
            if len(tail) > 40:
                del tail[:20]
    t = threading.Thread(target=drain, daemon=True)
    t.start()
    try:
        for line in proc.stdout:
            if job is not None:
                if job.cancelled:
                    proc.kill()
                    raise Cancelled()
                if line.startswith("out_time_us=") and duration:
                    try:
                        sec = int(line.split("=", 1)[1]) / 1e6
                    except ValueError:
                        continue
                    job.update(progress=min(int(sec), int(duration)), total=int(duration))
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
        t.join(timeout=5)
    if proc.returncode != 0:
        err = "".join(l for l in tail if "showinfo" not in l)[-800:].strip()
        raise RuntimeError(err or f"ffmpeg exited {proc.returncode}")


def grab_frame(path: str, t: float, width: int = 320) -> Image.Image:
    """Decode the single frame at time t (accurate seek), scaled to `width`."""
    for tt in (t, max(0.0, t - 0.5), 0.0):
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{tt:.3f}", "-i", path, "-frames:v", "1",
             "-vf", f"scale={width}:-2", "-f", "image2pipe", "-c:v", "png", "-"],
            capture_output=True, **_POPEN_KW)
        if proc.returncode == 0 and proc.stdout:
            img = Image.open(io.BytesIO(proc.stdout))
            img.load()
            return img.convert("RGB")
    raise RuntimeError(f"could not decode a frame at {t:.2f}s: "
                       + proc.stderr.decode(errors="replace").strip()[-200:])


def gray_sequence(path: str, fps: float = 1.0, size: int = 32, job=None,
                  duration: float = 0) -> np.ndarray:
    """Decode the whole video at `fps` as size x size grayscale -> (n, size, size) uint8."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-an",
         "-vf", f"fps={fps},scale={size}:{size}:flags=area,format=gray",
         "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_POPEN_KW)
    frame_bytes = size * size
    chunks, n = [], 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes * 16)
            if not buf:
                break
            chunks.append(buf)
            n += len(buf) // frame_bytes
            if job is not None:
                if job.cancelled:
                    proc.kill()
                    raise Cancelled()
        err = proc.stderr.read()
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0:
        raise RuntimeError(err.decode(errors="replace").strip()[-300:] or "ffmpeg failed")
    data = b"".join(chunks)
    usable = len(data) - len(data) % frame_bytes
    return np.frombuffer(data[:usable], dtype=np.uint8).reshape(-1, size, size)
