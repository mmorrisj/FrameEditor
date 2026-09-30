"""Mix: layer sounds from the sound library onto a video.

A mix is a video (a Colour match render of a chain, or any library video) plus
sound tracks. Each clip on a track is a sound placed at a time, with how much
of it plays, an offset into it, volume, fades and optional looping (for beds
that must run longer than the recording). The video's own audio is kept as
its own track with a volume and mute.

Nothing is changed until you render: the mix is a small JSON file. Rendering
mixes every clip with ffmpeg and puts the result on the video *without*
re-encoding the picture, so it is quick and loses no quality.

Snap points for placing sounds: the joins between segments of a chained
video, and scene cuts when a library video has been split into scenes.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import tempfile
import time

from .. import config, library, media
from . import colormatch, sounds

MID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
MAX_CLIPS = 400


def _root(*parts: str) -> str:
    return os.path.join(config.WORK, "mixes", *parts)


# ==== the video under the mix =========================================================================

def resolve_video(spec: dict) -> dict:
    """{path, name, markers} for {kind: library, id} or {kind: colormatch, session, file}."""
    kind = spec.get("kind")
    if kind == "library":
        v = library.require(spec.get("id") or "")
        markers = []
        try:  # scene cuts, if the video has been split into scenes
            from . import scenes
            if scenes.detected(v["id"]):
                markers = [{"t": s["start"], "label": f"scene {s['n']}"} for s in scenes.cuts(v["id"])["scenes"][1:]]
        except Exception:
            markers = []
        return {"path": v["path"], "name": v["name"], "markers": markers}
    if kind == "colormatch":
        s = colormatch.load(spec.get("session") or "")
        out = s.get("outputs") or {}
        name = spec.get("file") or out.get("joined")
        if not name:
            raise ValueError("that colour match session hasn't been rendered")
        path = colormatch.output_path(s["id"], name)
        if not os.path.isfile(path):
            raise LookupError("the rendered video is missing; render the session again")
        return {"path": path, "name": name, "markers": _joins(s)}
    raise ValueError("unknown video")


def _joins(s: dict) -> list[dict]:
    """Where each segment starts in a rendered chain (after dropped repeats and crossfades)."""
    out = s.get("outputs") or {}
    fps = colormatch.eval_frac((s.get("target") or {}).get("fps_frac") or "") or 16
    if out.get("ranges"):
        starts = [a for a, _ in out["ranges"]][1:]
        return [{"t": round((a - 1) / fps, 3), "label": f"segment {k + 2}"} for k, a in enumerate(starts)]
    t, marks = 0, []  # sessions rendered before ranges were recorded: rebuild from the segments
    xf = out.get("crossfade") or 0
    for k, sg in enumerate(s["segments"]):
        if k:
            marks.append({"t": round(t / fps, 3), "label": f"segment {k + 1}"})
        t += sg["frames"] - (sg.get("overlap") or 0 if k else 0) - (xf if k < len(s["segments"]) - 1 else 0)
    return marks


# ==== mixes ===========================================================================================

def _new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def create(spec: dict, name: str | None = None) -> dict:
    v = resolve_video(spec)
    info = media.probe(v["path"])
    mid = _new_id()
    m = {"id": mid, "name": name or os.path.splitext(v["name"])[0], "created": time.time(), "updated": time.time(),
         "video": {k: spec[k] for k in ("kind", "id", "session", "file") if spec.get(k)},
         "duration": round(info["duration"], 3), "fps": info.get("fps"), "has_audio": bool(info.get("has_audio")),
         "width": info["width"], "height": info["height"],
         "original": {"volume_db": 0.0, "mute": False},
         "tracks": [{"id": "t1", "name": "Track 1", "volume_db": 0.0, "mute": False},
                    {"id": "t2", "name": "Track 2", "volume_db": 0.0, "mute": False}],
         "clips": [], "outputs": []}
    os.makedirs(_root(mid), exist_ok=True)
    _write(m)
    return get(mid)


def _write(m: dict) -> None:
    with open(_root(m["id"], "mix.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)


def _read(mid: str) -> dict:
    if not MID_RE.match(mid or ""):
        raise ValueError("bad mix id")
    p = _root(mid, "mix.json")
    if not os.path.exists(p):
        raise LookupError("no such mix")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get(mid: str) -> dict:
    """The mix, with its video's markers and each clip's sound details for the editor."""
    m = _read(mid)
    try:
        v = resolve_video(m["video"])
        m["video_name"], m["markers"], m["video_ok"] = v["name"], v["markers"], True
    except (ValueError, LookupError) as e:
        m["video_name"], m["markers"], m["video_ok"], m["video_error"] = "", [], False, str(e)
    snd = {}
    for c in m["clips"]:
        if c["sound"] not in snd:
            try:
                x = sounds.get(c["sound"])
                snd[c["sound"]] = {k: x[k] for k in ("id", "name", "category", "duration", "envelope", "loop")}
            except LookupError:
                snd[c["sound"]] = None
    m["sounds"] = snd
    return m


def list_mixes() -> list[dict]:
    base = _root()
    out = []
    if os.path.isdir(base):
        for mid in sorted(os.listdir(base), reverse=True):
            if MID_RE.match(mid) and os.path.exists(_root(mid, "mix.json")):
                m = _read(mid)
                out.append({"id": mid, "name": m["name"], "updated": m["updated"], "clips": len(m["clips"]),
                            "duration": m["duration"], "outputs": len(m["outputs"])})
    return out


def _num(v, lo, hi, default=0.0) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def save(mid: str, data: dict) -> dict:
    """Replace the editable parts (name, original audio, tracks, clips), validated."""
    m = _read(mid)
    if "name" in data:
        m["name"] = str(data["name"] or m["name"])[:120]
    o = data.get("original") or m["original"]
    m["original"] = {"volume_db": _num(o.get("volume_db"), -60, 12), "mute": bool(o.get("mute"))}
    tracks = []
    for t in (data.get("tracks") or m["tracks"])[:32]:
        tid = str(t.get("id") or "")[:20]
        if tid and not any(x["id"] == tid for x in tracks):
            tracks.append({"id": tid, "name": str(t.get("name") or tid)[:60], "volume_db": _num(t.get("volume_db"), -60, 12),
                           "mute": bool(t.get("mute"))})
    if not tracks:
        raise ValueError("a mix needs at least one track")
    ids = {t["id"] for t in tracks}
    clips = []
    for c in (data.get("clips") if "clips" in data else m["clips"])[:MAX_CLIPS]:
        if c.get("track") not in ids:
            continue
        try:
            sid = int(c["sound"])
        except (KeyError, TypeError, ValueError):
            continue
        length = _num(c.get("length"), 0.02, 36000, 1.0)
        clips.append({"id": str(c.get("id") or secrets.token_hex(4))[:20], "track": c["track"], "sound": sid,
                      "start": _num(c.get("start"), 0, 36000), "offset": _num(c.get("offset"), 0, 36000),
                      "length": length, "loop": bool(c.get("loop")), "volume_db": _num(c.get("volume_db"), -60, 12),
                      "fade_in": _num(c.get("fade_in"), 0, length / 2), "fade_out": _num(c.get("fade_out"), 0, length / 2)})
    m["tracks"], m["clips"], m["updated"] = tracks, clips, time.time()
    _write(m)
    return get(mid)


def delete(mid: str) -> None:
    _read(mid)
    shutil.rmtree(_root(mid), ignore_errors=True)


def video_path(mid: str) -> str:
    return resolve_video(_read(mid)["video"])["path"]


def output_path(mid: str, name: str) -> str:
    _read(mid)
    if not re.match(r"^[^/\\]{1,200}$", name or "") or name in (".", "..") or name == "mix.json":
        raise ValueError("bad file name")
    return _root(mid, name)


# ==== rendering ========================================================================================

def _db(x: float) -> str:
    return f"{x:.2f}dB"


def build_graph(m: dict, sound_paths: dict[int, str]) -> tuple[list[str], str, int]:
    """(extra ffmpeg inputs, filter graph, number of mixed inputs). Input 0 is the video."""
    tracks = {t["id"]: t for t in m["tracks"]}
    inputs, chains, labels = [], [], []
    if m.get("has_audio") and not m["original"]["mute"]:
        chains.append(f"[0:a:0]aresample=48000,aformat=channel_layouts=stereo,volume={_db(m['original']['volume_db'])}[orig]")
        labels.append("[orig]")
    k = 1
    for c in m["clips"]:
        t = tracks.get(c["track"])
        if not t or t["mute"] or c["sound"] not in sound_paths or c["start"] >= m["duration"]:
            continue
        length = min(c["length"], m["duration"] - c["start"])
        inputs += (["-stream_loop", "-1"] if c["loop"] else []) + ["-i", sound_paths[c["sound"]]]
        f = [f"[{k}:a:0]aresample=48000", "aformat=channel_layouts=stereo",
             f"atrim=start={c['offset']:.4f}:duration={length:.4f}", "asetpts=PTS-STARTPTS"]
        if c["fade_in"] > 0:
            f.append(f"afade=t=in:st=0:d={c['fade_in']:.4f}")
        if c["fade_out"] > 0:
            f.append(f"afade=t=out:st={max(0.0, length - c['fade_out']):.4f}:d={c['fade_out']:.4f}")
        f.append(f"volume={_db(t['volume_db'] + c['volume_db'])}")
        f.append(f"adelay=delays={int(round(c['start'] * 1000))}:all=1")
        chains.append(",".join(f) + f"[c{k}]")
        labels.append(f"[c{k}]")
        k += 1
    if not labels:
        raise ValueError("nothing to mix: add sounds, or unmute a track or the original audio")
    d = m["duration"]
    chains.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0,"
                  f"apad=whole_dur={d:.4f},atrim=duration={d:.4f},alimiter=limit=0.98:level=false[mix]")
    return inputs, ";\n".join(chains), len(labels)


def render(job, mid: str, to_library: bool = False) -> dict:
    m = _read(mid)
    src = resolve_video(m["video"])["path"]
    paths = {}
    for c in m["clips"]:
        if c["sound"] not in paths:
            try:
                p = sounds.get(c["sound"])["path"]
            except LookupError:
                continue
            if os.path.exists(p):
                paths[c["sound"]] = p
    inputs, graph, n = build_graph(m, paths)
    ext = os.path.splitext(src)[1].lower()
    ext = ext if ext in (".mp4", ".mov", ".mkv", ".m4v") else ".mp4"
    safe = re.sub(r"[^\w\- .()]+", "_", m["name"]).strip(" ._") or "mix"
    from ..fileops import _free
    out = _free(_root(mid, f"{safe}-mixed{ext}"))
    with tempfile.TemporaryDirectory() as tmp:
        gpath = os.path.join(tmp, "graph.txt")
        with open(gpath, "w", encoding="utf-8") as f:
            f.write(graph)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, *inputs, "-/filter_complex", gpath,
               "-map", "0:v:0", "-map", "[mix]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        if ext in (".mp4", ".mov", ".m4v"):
            cmd += ["-movflags", "+faststart"]
        if job:
            job.update(message=f"mixing {n} layer{'s' if n != 1 else ''}", total=int(m["duration"]))
        media.run(cmd + [out], job=job, duration=m["duration"])
    m = _read(mid)
    entry = {"file": os.path.basename(out), "rendered": time.time(), "layers": n}
    if to_library:  # the mixes folder acts as its root, like uploads do
        try:
            entry["library_id"] = library.add_path(out, root=_root())["id"]
        except (OSError, ValueError):
            entry["library_id"] = None
    m["outputs"] = [entry] + m["outputs"][:9]
    _write(m)
    return {"mix": mid, "file": entry["file"], "layers": n}
