"""Split a video into one clip per scene.

detect() decodes the video once (downscaled) and records ffmpeg's scene-change
score for every frame, cached until the file changes. Cuts are then derived
from those scores instantly, so the sensitivity can be tuned freely:

  threshold  a frame starts a new scene when its score is above this (0-1)
  min_len    a cut closer than this many seconds to the previous cut is
             ignored, so flashes and strobes don't create slivers

Manual edits sit on top: cuts added at a time, and detected cuts removed.
They are kept per video in work/videos/<id>/scenes/state.json.

Export modes:
  exact  frame-accurate: every clip starts on its cut frame. Re-encodes
         (H.264, high quality) with the audio as AAC.
  fast   no re-encode, near-instant, but each cut moves to the next
         keyframe, so clips can start up to a few seconds late.
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import shutil
import time
import zipfile

import numpy as np

from .. import config, library, media
from ..db import cache_get, cache_put

DEFAULTS = {"threshold": 0.3, "min_len": 1.0}
QUALITY_CRF = 18


def _dir(vid: str, *parts: str) -> str:
    return library.work_dir(vid, "scenes", *parts)


# --- detection ------------------------------------------------------------------------

def detect(job, vid: str) -> dict:
    """Per-frame scene scores for a library video (cached)."""
    v = library.require(vid)
    key = library.cache_key(v, "sc1")
    raw = cache_get(vid, "scenes", key)
    if raw:
        z = np.load(io.BytesIO(raw))
        return {"times": z["t"], "scores": z["s"]}
    times, scores = [], []
    pending = {}
    pat_t = re.compile(r"Parsed_metadata.*\bpts_time:([-\d.eE+]+)")
    pat_s = re.compile(r"Parsed_metadata.*lavfi\.scene_score=([\d.eE+-]+)")

    def on_line(line):
        m = pat_t.search(line)
        if m:
            pending["t"] = float(m.group(1))
            return
        m = pat_s.search(line)
        if m and "t" in pending:
            times.append(pending.pop("t"))
            scores.append(float(m.group(1)))
    if job:
        job.update(message="finding scene changes")
    media.run(["ffmpeg", "-hide_banner", "-loglevel", "info", "-i", v["path"], "-an", "-sn",
               "-vf", "scale=320:-2,select='gte(scene\\,0)',metadata=print:key=lavfi.scene_score",
               "-f", "null", "-"], job=job, duration=v["info"].get("duration") or 0, on_stderr=on_line)
    t, s = np.array(times, np.float64), np.array(scores, np.float32)
    buf = io.BytesIO()
    np.savez(buf, t=t, s=s)
    cache_put(vid, "scenes", key, buf.getvalue())
    return {"times": t, "scores": s}


def detected(vid: str) -> bool:
    v = library.get(vid)
    return bool(v and cache_get(vid, "scenes", library.cache_key(v, "sc1")))


# --- state and cuts ------------------------------------------------------------------------

def load_state(vid: str) -> dict:
    p = _dir(vid, "state.json")
    s = dict(DEFAULTS, added=[], removed=[])
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            s.update(json.load(f))
    return s


def save_state(vid: str, **kw) -> dict:
    s = load_state(vid)
    if kw.get("threshold") is not None:
        t = float(kw["threshold"])
        if not 0.01 <= t <= 1:
            raise ValueError("threshold must be between 0.01 and 1")
        s["threshold"] = t
    if kw.get("min_len") is not None:
        m = float(kw["min_len"])
        if not 0 <= m <= 60:
            raise ValueError("minimum length must be between 0 and 60 seconds")
        s["min_len"] = m
    _write(vid, s)
    return s


def _near(t: float, ts: list[float], tol: float) -> bool:
    return any(abs(t - x) <= tol for x in ts)


def cuts(vid: str, state: dict | None = None) -> dict:
    """Scene list from the cached scores plus manual edits."""
    v = library.require(vid)
    state = state or load_state(vid)
    sc = detect(None, vid) if detected(vid) else {"times": np.zeros(0), "scores": np.zeros(0)}
    dur = float(v["info"].get("duration") or (sc["times"][-1] if len(sc["times"]) else 0))
    fps = v["info"].get("fps") or 25
    tol = 0.6 / fps
    auto = [float(t) for t, s in zip(sc["times"], sc["scores"]) if s > state["threshold"] and t > tol]
    auto = [t for t in auto if not _near(t, state["removed"], tol)]
    added = [t for t in state["added"] if tol < t < dur - tol]
    out, last = [], 0.0
    for t in sorted(set(auto) | set(added)):
        manual = _near(t, added, tol)
        if manual or (t - last >= state["min_len"]):
            out.append(t)
            last = t
    while out and not _near(out[-1], added, tol) and dur - out[-1] < state["min_len"]:
        out.pop()  # a sliver at the very end joins the scene before it
    bounds = [0.0] + out + [dur]
    scenes = [{"n": i + 1, "start": round(bounds[i], 4), "end": round(bounds[i + 1], 4),
               "length": round(bounds[i + 1] - bounds[i], 3),
               "manual": i > 0 and _near(bounds[i], added, tol)}
              for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > tol]
    peaks = sorted(((float(t), float(s)) for t, s in zip(sc["times"], sc["scores"])), key=lambda x: -x[1])[:400]
    return {"video_id": vid, "duration": dur, "fps": fps, "detected": bool(len(sc["times"])),
            "threshold": state["threshold"], "min_len": state["min_len"], "scenes": scenes,
            "peaks": sorted(peaks)}


def add_cut(vid: str, t: float) -> dict:
    s = load_state(vid)
    t = round(float(t), 4)
    s["removed"] = [x for x in s["removed"] if abs(x - t) > 0.02]
    if t not in s["added"]:
        s["added"].append(t)
    _write(vid, s)
    return cuts(vid, s)


def remove_cut(vid: str, t: float) -> dict:
    """Merge the scene starting at t into the previous one."""
    s = load_state(vid)
    t = float(t)
    s["added"] = [x for x in s["added"] if abs(x - t) > 0.02]
    s["removed"].append(round(t, 4))
    _write(vid, s)
    return cuts(vid, s)


def reset_edits(vid: str) -> dict:
    s = load_state(vid)
    s["added"], s["removed"] = [], []
    _write(vid, s)
    return cuts(vid, s)


def _write(vid: str, s: dict) -> None:
    os.makedirs(_dir(vid), exist_ok=True)
    with open(_dir(vid, "state.json"), "w", encoding="utf-8") as f:
        json.dump({k: s[k] for k in ("threshold", "min_len", "added", "removed")}, f)


def thumb(vid: str, t: float) -> str:
    """Cached 240px frame at time t."""
    v = library.require(vid)
    p = _dir(vid, "thumbs", f"{int(round(t * 1000)):09d}.jpg")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        media.grab_frame(v["path"], t, 240).save(p, quality=82)
    return p


# --- export ----------------------------------------------------------------------------------

def split(job, src: str, out_dir: str, scenes: list[dict], mode: str = "exact",
          info: dict | None = None, name: str | None = None, keep: set[int] | None = None) -> dict:
    """Write one clip per scene into out_dir, plus scenes.json.

    `scenes` must cover the video in order. `keep` limits which scene numbers
    are written (fast mode still cuts every scene in its single pass, then
    discards the rest).
    """
    if mode not in ("exact", "fast"):
        raise ValueError("mode must be exact or fast")
    if not scenes:
        raise ValueError("no scenes to export")
    info = info or media.probe(src)
    fps = info.get("fps") or 25
    stem = os.path.splitext(name or os.path.basename(src))[0]
    os.makedirs(out_dir, exist_ok=True)
    clips = []
    if mode == "fast":
        # one pass over the whole video; the segment muxer cuts at the first
        # keyframe after each cut, so a clip never shows the previous scene
        pattern = os.path.join(out_dir, ".part_%03d.mp4")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-map", "0:v:0", "-map", "0:a?",
               "-c", "copy", "-f", "segment", "-reset_timestamps", "1"]
        if len(scenes) > 1:
            cmd += ["-segment_times", ",".join(f"{s['start']:.4f}" for s in scenes[1:])]
        if job:
            job.update(message="cutting (no re-encode)")
        media.run(cmd + [pattern], job=job, duration=scenes[-1]["end"])
        for i, s in enumerate(scenes):
            p = pattern % i
            if not os.path.exists(p):
                continue
            if keep and s["n"] not in keep:
                os.remove(p)
                continue
            target = os.path.join(out_dir, f"{stem}_scene_{s['n']:03d}.mp4")
            os.replace(p, target)
            clips.append({**s, "file": os.path.basename(target), "size": os.path.getsize(target)})
        for f in os.listdir(out_dir):  # segments beyond the scene list (shouldn't happen)
            if f.startswith(".part_"):
                os.remove(os.path.join(out_dir, f))
    else:
        scenes = [s for s in scenes if not keep or s["n"] in keep]
        total = sum(s["end"] - s["start"] for s in scenes)
        done = 0.0
        for k, s in enumerate(scenes, 1):
            if job:
                job.check()
                job.update(message=f"scene {k} of {len(scenes)}", progress=int(done), total=int(total) or 1)
            target = os.path.join(out_dir, f"{stem}_scene_{s['n']:03d}.mp4")
            length = max(0.5 / fps, s["end"] - s["start"] - 0.25 / fps)  # stop just before the next cut
            media.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s['start']:.4f}", "-i", src,
                       "-t", f"{length:.4f}", "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                       "-preset", "medium", "-crf", str(QUALITY_CRF), "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", target])
            clips.append({**s, "file": os.path.basename(target), "size": os.path.getsize(target)})
            done += s["end"] - s["start"]
    manifest = {"source": src, "mode": mode, "created": time.time(), "clips": clips}
    with open(os.path.join(out_dir, "scenes.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return {"clips": len(clips), "mode": mode, "folder": out_dir}


def export(job, vid: str, mode: str = "exact", only: list[int] | None = None,
           to_library: bool = False) -> dict:
    v = library.require(vid)
    c = cuts(vid)
    keep = set(only) if only else None
    if keep and not keep & {s["n"] for s in c["scenes"]}:
        raise ValueError("none of the chosen scenes exist")
    run = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    out = _dir(vid, "exports", run)
    try:
        res = split(job, v["path"], out, c["scenes"], mode=mode, info=v["info"], name=v["name"], keep=keep)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    res["run"] = run
    if to_library:  # clips land in uploads/<video>-scenes/ where every tool can use them
        from .. import fileops
        dest = fileops._free(os.path.join(config.UPLOADS, f"{os.path.splitext(v['name'])[0]}-scenes"))
        shutil.copytree(out, dest, ignore=shutil.ignore_patterns("*.json"))
        added = 0
        for f in sorted(os.listdir(dest)):
            library.add_path(os.path.join(dest, f))
            added += 1
        res["library_folder"] = dest
        res["added"] = added
    return res


def list_exports(vid: str) -> list[dict]:
    base = _dir(vid, "exports")
    out = []
    if os.path.isdir(base):
        for run in sorted(os.listdir(base), reverse=True):
            p = os.path.join(base, run, "scenes.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)
                out.append({"run": run, "mode": m["mode"], "created": m["created"],
                            "clips": [{"file": c["file"], "n": c["n"], "start": c["start"], "end": c["end"],
                                       "size": c["size"]} for c in m["clips"]]})
    return out


def export_path(vid: str, run: str, name: str | None = None) -> str:
    if not re.match(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$", run):
        raise ValueError("bad export id")
    d = _dir(vid, "exports", run)
    if name is None:
        return d
    if os.path.basename(name) != name or name.startswith("."):
        raise ValueError("bad file name")
    return os.path.join(d, name)


def zip_export(vid: str, run: str) -> str:
    d = export_path(vid, run)
    if not os.path.isdir(d):
        raise LookupError("no such export")
    v = library.require(vid)
    os.makedirs(config.EXPORTS, exist_ok=True)
    out = os.path.join(config.EXPORTS, f"{os.path.splitext(v['name'])[0]}-scenes-{run}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(os.listdir(d)):
            z.write(os.path.join(d, f), f)
    return out


def delete_export(vid: str, run: str) -> None:
    d = export_path(vid, run)
    shutil.rmtree(d, ignore_errors=True)


def split_video(job, vid: str, mode: str = "exact", to_library: bool = False) -> dict:
    """Detect (if needed) and export every scene with the video's current settings."""
    detect(job, vid)
    return export(job, vid, mode=mode, to_library=to_library)
