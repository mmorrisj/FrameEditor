"""ffmpeg/ffprobe plumbing: probe, extract-to-frames, repackage, job tracking.

All the video I/O lives here, no web. Each project is a directory under
projects/<id>/ containing:
    original.<ext>   the upload, untouched (also the audio source at repackage)
    frames/f%06d.png full-resolution frames, the editable working set
    thumbs/t%06d.jpg 200px-wide thumbnails for the filmstrip UI
    pristine/        pre-edit copies of any frame that has been modified
    meta.json        probe results + frame count, written after extraction
    output.mp4       the repackaged result

Extraction normalizes to constant frame rate (fps=<avg_frame_rate> filter):
AI video tools sometimes emit VFR, and extracting VFR frames then re-encoding
at a fixed rate would drift the audio out of sync.
"""
from __future__ import annotations

import glob
import json
import os
import re
import secrets
import shutil
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTS = os.path.join(HERE, "projects")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _set(pid: str, **kw) -> None:
    with _lock:
        _jobs.setdefault(pid, {}).update(kw)


def _job(pid: str) -> dict:
    with _lock:
        return dict(_jobs.get(pid) or {"state": "new", "progress": 0, "total": 0, "error": None})


def project_dir(pid: str) -> str:
    return os.path.join(PROJECTS, pid)


def _original(d: str) -> str:
    hits = glob.glob(os.path.join(d, "original.*"))
    if not hits:
        raise FileNotFoundError("no original video in project")
    return hits[0]


def probe(path: str) -> dict:
    """Video stream geometry/rate, duration, and whether an audio stream exists."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,avg_frame_rate,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    vid = next(s for s in data["streams"] if s.get("codec_type") == "video")
    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])

    def _frac(s):
        num, _, den = (s or "0/0").partition("/")
        num, den = float(num or 0), float(den or 1)
        return num / den if den and num else 0.0

    fps_frac = vid.get("avg_frame_rate")
    if not _frac(fps_frac):                       # some containers leave avg empty
        fps_frac = vid.get("r_frame_rate")
    fps = _frac(fps_frac)
    if not fps:
        raise RuntimeError("could not determine frame rate")
    return {
        "width": vid["width"], "height": vid["height"],
        "fps_frac": fps_frac, "fps": round(fps, 3),
        "duration": float(data["format"].get("duration") or 0),
        "has_audio": has_audio,
    }


def _run_progress(pid: str, cmd: list[str], total: int) -> None:
    """Run ffmpeg, updating the job from its -progress key=value stream.

    stderr is capped at -loglevel error so it stays within the pipe buffer
    while we consume stdout; we only read it after exit, for the error text.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        m = re.match(r"frame=(\d+)", line.strip())
        if m:
            _set(pid, progress=int(m.group(1)), total=total)
    proc.wait()
    if proc.returncode != 0:
        err = (proc.stderr.read() or "")[-800:].strip()
        raise RuntimeError(err or f"ffmpeg exited {proc.returncode}")


# --- projects ---------------------------------------------------------------

def create_project(file_storage) -> str:
    name, ext = os.path.splitext(file_storage.filename)
    ext = ext.lower()
    if ext not in VIDEO_EXTS:
        raise ValueError(f"unsupported extension {ext!r}")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "video"
    pid = f"{slug}-{secrets.token_hex(3)}"
    d = project_dir(pid)
    os.makedirs(d)
    file_storage.save(os.path.join(d, "original" + ext))
    return pid


def list_projects() -> list[dict]:
    out = []
    if not os.path.isdir(PROJECTS):
        return out
    for pid in os.listdir(PROJECTS):
        d = project_dir(pid)
        if not os.path.isdir(d):
            continue
        entry = {"id": pid, "mtime": os.path.getmtime(d), **_job(pid)}
        meta_path = os.path.join(d, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                entry.update(json.load(f))
            if entry["state"] == "new":      # extracted in a previous run
                entry["state"] = "ready"
        entry["output"] = os.path.exists(os.path.join(d, "output.mp4"))
        out.append(entry)
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


def delete_project(pid: str) -> None:
    shutil.rmtree(project_dir(pid))
    with _lock:
        _jobs.pop(pid, None)


def meta(pid: str) -> dict:
    with open(os.path.join(project_dir(pid), "meta.json")) as f:
        return json.load(f)


def status(pid: str) -> dict:
    d = project_dir(pid)
    s = {"id": pid, **_job(pid)}
    meta_path = os.path.join(d, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            s.update(json.load(f))
        if s["state"] == "new":
            s["state"] = "ready"
    pristine = os.path.join(d, "pristine")
    s["edited"] = sorted(
        int(f[1:7]) for f in os.listdir(pristine)) if os.path.isdir(pristine) else []
    s["output"] = os.path.exists(os.path.join(d, "output.mp4"))
    return s


# --- jobs -------------------------------------------------------------------

def _extract(pid: str) -> None:
    d = project_dir(pid)
    try:
        src = _original(d)
        info = probe(src)
        total = int(info["duration"] * info["fps"])
        _set(pid, state="extracting", progress=0, total=total, error=None)
        frames, thumbs = os.path.join(d, "frames"), os.path.join(d, "thumbs")
        os.makedirs(frames, exist_ok=True)
        os.makedirs(thumbs, exist_ok=True)
        # one decode pass -> full frames + filmstrip thumbnails
        fc = (f"[0:v]fps={info['fps_frac']},split=2[f][t];"
              f"[t]scale=200:-2[ts]")
        _run_progress(pid, [
            "ffmpeg", "-y", "-loglevel", "error", "-progress", "pipe:1", "-nostats",
            "-i", src, "-filter_complex", fc,
            "-map", "[f]", os.path.join(frames, "f%06d.png"),
            "-map", "[ts]", "-q:v", "5", os.path.join(thumbs, "t%06d.jpg"),
        ], total)
        count = len(glob.glob(os.path.join(frames, "f*.png")))
        if not count:
            raise RuntimeError("extraction produced no frames")
        info["frames"] = count
        info["name"] = os.path.basename(src)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(info, f)
        _set(pid, state="ready", progress=count, total=count)
    except Exception as e:  # surfaced in /status, never kills the server
        _set(pid, state="error", error=str(e))


def _repackage(pid: str) -> None:
    d = project_dir(pid)
    try:
        info = meta(pid)
        _set(pid, state="packaging", progress=0, total=info["frames"], error=None)
        out = os.path.join(d, "output.mp4")
        if os.path.exists(out):
            os.remove(out)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-progress", "pipe:1", "-nostats",
               "-framerate", info["fps_frac"],
               "-i", os.path.join(d, "frames", "f%06d.png")]
        if info["has_audio"]:
            cmd += ["-i", _original(d), "-map", "0:v", "-map", "1:a", "-c:a", "copy"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        _run_progress(pid, cmd, info["frames"])
        _set(pid, state="ready", progress=info["frames"])
    except Exception as e:
        _set(pid, state="error", error=str(e))


def start_extract(pid: str) -> None:
    threading.Thread(target=_extract, args=(pid,), daemon=True).start()


def start_repackage(pid: str) -> None:
    threading.Thread(target=_repackage, args=(pid,), daemon=True).start()
