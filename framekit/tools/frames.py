"""Frame extraction: break a video into image files.

Modes (value meaning in brackets):
  all        every frame, normalized to the video's constant frame rate
  nth        every Nth frame [N, default 10]
  fps        a fixed number of frames per second [rate, default 1]
  scene      the first frame plus every scene change [threshold 0-1, default 0.3]
  keyframes  only the encoder's keyframes (fastest; decodes nothing else)

Each extraction is a "run": a folder holding f000001.<fmt> ..., 240px thumbnails
and manifest.json, which records every frame's source timestamp. Library videos
keep runs under work/videos/<id>/frames/<run>/; the CLI can target any folder.
"""
from __future__ import annotations

import glob
import json
import os
import re
import secrets
import shutil
import time
import zipfile

from .. import config, library, media

MODES = {
    "all": "Every frame",
    "nth": "Every Nth frame",
    "fps": "Frames per second",
    "scene": "Scene changes",
    "keyframes": "Keyframes only",
}
DEFAULT_VALUES = {"nth": 10, "fps": 1, "scene": 0.3}
FORMATS = ("jpg", "png", "webp")
THUMB_W = 240
RUN_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _filters(mode: str, value, info: dict) -> tuple[list[str], list[str]]:
    """(input options, filter list) for a mode."""
    if mode == "all":
        return [], ([f"fps={info['fps_frac']}"] if info.get("fps_frac") else [])
    if mode == "nth":
        n = int(value) if value not in (None, "") else DEFAULT_VALUES["nth"]
        if n < 1:
            raise ValueError("N must be at least 1")
        return [], [f"select='not(mod(n\\,{n}))'"]
    if mode == "fps":
        r = float(value) if value not in (None, "") else DEFAULT_VALUES["fps"]
        if not 0 < r <= 240:
            raise ValueError("rate must be between 0 and 240")
        return [], [f"fps={r:g}"]
    if mode == "scene":
        th = float(value if value not in (None, "") else DEFAULT_VALUES["scene"])
        if not 0 < th < 1:
            raise ValueError("scene threshold must be between 0 and 1")
        return [], [f"select='eq(n\\,0)+gt(scene\\,{th:g})'"]
    if mode == "keyframes":
        return ["-skip_frame", "nokey"], []
    raise ValueError(f"unknown mode {mode!r}")


def _format_args(fmt: str) -> list[str]:
    if fmt == "jpg":
        return ["-q:v", "2"]
    if fmt == "png":
        return []
    if fmt == "webp":
        return ["-c:v", "libwebp", "-quality", "90"]
    raise ValueError(f"unknown format {fmt!r}")


def extract(job, src: str, out_dir: str, mode: str = "scene", value=None, fmt: str = "jpg",
            width: int | None = None, limit: int | None = None, thumbs: bool = True,
            info: dict | None = None, extra: dict | None = None) -> dict:
    """Extract frames from `src` into `out_dir` and write manifest.json."""
    info = info or media.probe(src)
    pre, vf = _filters(mode, value, info)
    fmt_args = _format_args(fmt)
    if width:
        vf.append(f"scale={int(width)}:-2")
    vf.append("showinfo")  # logs each output frame's pts_time -> manifest timestamps
    os.makedirs(out_dir, exist_ok=True)
    if thumbs:
        os.makedirs(os.path.join(out_dir, "thumbs"), exist_ok=True)
        graph = "[0:v]" + ",".join(vf) + f",split=2[f][t];[t]scale={THUMB_W}:-2[ts]"
    else:
        graph = "[0:v]" + ",".join(vf) + "[f]"
    per_out = media.vfr_args() + (["-frames:v", str(int(limit))] if limit else [])
    cmd = ["ffmpeg", "-y", "-loglevel", "info", "-hide_banner", *pre, "-i", src,
           "-filter_complex", graph,
           "-map", "[f]", *per_out, *fmt_args, os.path.join(out_dir, f"f%06d.{fmt}")]
    if thumbs:
        cmd += ["-map", "[ts]", *per_out, "-q:v", "5", os.path.join(out_dir, "thumbs", "t%06d.jpg")]

    times: list[float] = []
    pat = re.compile(r"Parsed_showinfo.*\bn:\s*(\d+).*?pts_time:\s*([-\d.eE+]+)")

    def on_line(line: str):
        m = pat.search(line)
        if m:
            times.append(float(m.group(2)))
    if job:
        job.update(message=f"extracting ({MODES[mode].lower()})")
    started = time.time()
    media.run(cmd, job=job, duration=info.get("duration") or 0, on_stderr=on_line)

    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out_dir, f"f*.{fmt}")))
    if not files:
        raise RuntimeError("no frames were extracted (try a lower scene threshold or another mode)")
    frames = [{"n": i + 1, "file": f, "t": round(times[i], 4) if i < len(times) else None}
              for i, f in enumerate(files)]
    manifest = {
        "id": os.path.basename(os.path.normpath(out_dir)),
        "source": os.path.abspath(src), "name": os.path.basename(src),
        "mode": mode, "value": value if mode != "all" and mode != "keyframes" else None,
        "fmt": fmt, "width": width, "limit": limit, "thumbs": thumbs,
        "created": time.time(), "seconds": round(time.time() - started, 2),
        "count": len(frames), "video": {k: info.get(k) for k in ("width", "height", "fps", "duration")},
        "frames": frames, **(extra or {}),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    if job:
        job.update(message=f"{len(frames)} frames", progress=job.total or 1, total=job.total or 1)
    return {k: v for k, v in manifest.items() if k != "frames"}


# --- runs for library videos ---------------------------------------------------

def new_run_id(mode: str) -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + mode + "-" + secrets.token_hex(2)


def run_dir(vid: str, run: str) -> str:
    if not RUN_RE.match(run):
        raise ValueError("bad run id")
    return library.work_dir(vid, "frames", run)


def extract_video(job, vid: str, mode: str = "scene", value=None, fmt: str = "jpg",
                  width: int | None = None, limit: int | None = None) -> dict:
    v = library.require(vid)
    run = new_run_id(mode)
    out = run_dir(vid, run)
    try:
        return extract(job, v["path"], out, mode=mode, value=value, fmt=fmt, width=width,
                       limit=limit, info=v["info"], extra={"video_id": vid})
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def list_runs(vid: str) -> list[dict]:
    base = library.work_dir(vid, "frames")
    out = []
    for m in glob.glob(os.path.join(base, "*", "manifest.json")):
        with open(m, encoding="utf-8") as f:
            d = json.load(f)
        d.pop("frames", None)
        d["analyzed"] = os.path.exists(os.path.join(os.path.dirname(m), "analysis", "state.json"))
        out.append(d)
    return sorted(out, key=lambda d: d["created"], reverse=True)


def get_run(vid: str, run: str) -> dict:
    p = os.path.join(run_dir(vid, run), "manifest.json")
    if not os.path.exists(p):
        raise LookupError("no such frame run")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def delete_run(vid: str, run: str) -> None:
    d = run_dir(vid, run)
    if os.path.isdir(d):
        shutil.rmtree(d)


def zip_run(vid: str, run: str) -> str:
    """Zip a run's full-size frames (stored, not recompressed) into exports/."""
    m = get_run(vid, run)
    d = run_dir(vid, run)
    stem = os.path.splitext(m["name"])[0]
    os.makedirs(config.EXPORTS, exist_ok=True)
    out = os.path.join(config.EXPORTS, f"{stem}-{run}-frames.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for fr in m["frames"]:
            z.write(os.path.join(d, fr["file"]), fr["file"])
        z.writestr("manifest.json", json.dumps(m, indent=1))
    return out
