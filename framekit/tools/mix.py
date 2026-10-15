"""Mix: splice videos together and layer sounds from the sound library on top.

A mix has a video track and sound tracks.

The video track is a row of clips, each a piece (in to out) of a library video
or a Colour match render. Clips play one after another; the join into each clip
is a straight cut, a crossfade, or a fade through black, and a transition
overlaps the two clips by its length. Each clip's own audio comes along, with a
volume and mute per clip and a level for the whole video track ("original").
Everything is placed on the output frame grid, so the preview, the audio and
the rendered picture agree to the frame.

Output settings: size (auto = the first clip's), frame rate (auto = the rate
most clips already have) and how clips of another shape are fitted (black bars,
a blurred fill, or cropped to fill).

Each sound clip on a track is a sound placed at a time, with how much of it
plays, an offset into it, volume, fades and optional looping (for beds that
must run longer than the recording).

Nothing is changed until you render: the mix is a small JSON file. A video
track that is a single whole clip at its own size and rate is copied without
re-encoding; anything spliced, trimmed or converted is encoded once (x264,
CRF 18). Rendering is always exact, so sounds stay where you put them.

Snap points: the joins between clips, the joins between segments of a chained
video, and scene cuts when a library video has been split into scenes.

Cleanup and levels:
  per clip    match loudness (every clip to a common perceived loudness, so
              sounds from different sources sit together), noise reduction,
              low cut, high cut, de-click, and pan. The cleanup is applied once
              with ffmpeg and cached; the preview and the render both play that
              processed copy, so what you hear is what renders.
  per track   ducking: the track dips by N dB while the video's own audio (or
              another track) is playing. The dip is computed as one gain curve
              that the preview and the render both follow.
  master      glue compression, a loudness target (streaming -14, web -16,
              broadcast -23 LUFS; two-pass loudnorm) and a true-peak limiter.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np

from .. import config, library, media
from . import colormatch, sounds

MID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
MAX_CLIPS = 400
MAX_VCLIPS = 200


def _root(*parts: str) -> str:
    return os.path.join(config.WORK, "mixes", *parts)


def _round(x: float) -> int:
    """Round half up (the browser's Math.round), so both sides place clips on the same frames."""
    return int(math.floor(x + 0.5))


# ==== video sources ===================================================================================

def norm_src(spec: dict, fill: bool = False) -> dict:
    """A source spec in canonical form: {kind: library, id} or {kind: colormatch, session, file}.
    fill: pin a colour match source to the session's current joined render."""
    spec = spec or {}
    if spec.get("kind") == "library":
        vid = str(spec.get("id") or "")
        if not re.match(r"^[\w\-]{1,64}$", vid):
            raise ValueError("bad video id")
        return {"kind": "library", "id": vid}
    if spec.get("kind") == "colormatch":
        sid, name = str(spec.get("session") or ""), spec.get("file")
        if not re.match(r"^[\w\-]{1,64}$", sid):
            raise ValueError("bad session id")
        if not name and fill:
            name = (colormatch.load(sid).get("outputs") or {}).get("joined")
        out = {"kind": "colormatch", "session": sid}
        if name:
            if not re.match(r"^[^/\\]{1,200}$", str(name)) or name in (".", ".."):
                raise ValueError("bad file name")
            out["file"] = str(name)
        return out
    raise ValueError("unknown video")


def skey(spec: dict) -> str:
    return hashlib.sha1(json.dumps(norm_src(spec), sort_keys=True).encode()).hexdigest()[:12]


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


_INFO: dict[tuple, dict] = {}


def source(spec: dict) -> dict:
    """A source with what the editor and the render need: path, name, markers, duration, rate,
    display size (pixel shape and rotation applied) and whether it has audio. Probes are cached."""
    v = resolve_video(spec)
    st = os.stat(v["path"])
    key = (v["path"], st.st_size, st.st_mtime_ns)
    info = _INFO.get(key)
    if info is None:
        info = _INFO[key] = media.probe(v["path"])
    w, h = info["width"], info["height"]
    sar = info.get("sar") or 1.0
    if abs(sar - 1) > 0.01:
        w = int(round(w * sar))
    if info.get("rotation") in (90, 270):
        w, h = h, w
    return {**v, "duration": round(info["duration"], 3), "fps": info.get("fps") or 25,
            "fps_frac": info.get("fps_frac") or "25/1", "width": w, "height": h, "sar": sar,
            "rotation": info.get("rotation") or 0, "has_audio": bool(info.get("has_audio")),
            "vcodec": info.get("vcodec"), "ext": os.path.splitext(v["path"])[1].lower()}


# ==== output settings and placement ======================================================================

SETTINGS_DEFAULT = {"size": "auto", "fps": "auto", "fit": "pad"}
SIZES = ["auto", "1920x1080", "1280x720", "832x480", "1080x1920", "720x1280", "480x832", "1080x1080", "720x720"]
FPS_CHOICES = {"auto": None, "12": "12/1", "15": "15/1", "16": "16/1", "23.976": "24000/1001", "24": "24/1",
               "25": "25/1", "29.97": "30000/1001", "30": "30/1", "50": "50/1", "60": "60/1"}
FITS = {"pad": "black bars", "blur": "blurred fill", "crop": "crop to fill"}
TRANSITIONS = {"cut": "cut", "crossfade": "crossfade", "fadeblack": "fade through black"}


def output_format(m: dict, srcs: dict) -> dict:
    """The resolved output: width, height, fps (fraction and number), fit."""
    st = {**SETTINGS_DEFAULT, **(m.get("settings") or {})}
    infos = [srcs[v["skey"]] for v in m["vclips"] if srcs.get(v["skey"], {}).get("ok")]
    if st["size"] != "auto":
        w, h = (int(x) for x in st["size"].split("x"))
    elif infos:
        w, h = infos[0]["width"], infos[0]["height"]
    else:
        w, h = 1280, 720
    w, h = max(2, w - w % 2), max(2, h - h % 2)
    frac = FPS_CHOICES.get(st["fps"]) or colormatch.pick_fps(infos, "auto")
    return {"width": w, "height": h, "fps_frac": frac, "fps": colormatch.eval_frac(frac), "fit": st["fit"],
            "auto_size": st["size"] == "auto", "auto_fps": st["fps"] == "auto"}


def layout(vclips: list[dict], fps: float) -> tuple[list[dict], float]:
    """Where each video clip sits, in whole output frames. A transition into a clip overlaps it
    with the one before by its length, at most half of either clip. (The page does the same.)"""
    rows, t, prev = [], 0, 0
    for i, v in enumerate(vclips):
        n = max(1, _round((v["out"] - v["in"]) * fps))
        tr = v.get("transition") or {}
        d = 0
        if i and tr.get("type") in ("crossfade", "fadeblack"):
            d = max(0, min(_round(tr.get("dur", 0.5) * fps), prev // 2, n // 2))
        start = t - d
        rows.append({"id": v["id"], "frames": n, "dframes": d, "sframe": start,
                     "start": start / fps, "len": n / fps, "overlap": d / fps})
        t, prev = start + n, n
    return rows, t / fps


# ==== mixes ===========================================================================================

def _new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def _vclip(spec: dict, s: dict, vid: str = "v1") -> dict:
    return {"id": vid, "src": spec, "in": 0.0, "out": s["duration"],
            "transition": {"type": "cut", "dur": 0.5}, "volume_db": 0.0, "mute": False}


def create(spec: dict | None = None, name: str | None = None) -> dict:
    """A new mix, starting from one video (or empty, to build up from the Videos list)."""
    vcl = []
    if spec:
        spec = norm_src(spec, fill=True)
        s = source(spec)
        vcl = [_vclip(spec, s)]
        name = name or os.path.splitext(s["name"])[0]
    mid = _new_id()
    m = {"id": mid, "name": name or "Untitled mix", "created": time.time(), "updated": time.time(),
         "vclips": vcl, "settings": dict(SETTINGS_DEFAULT),
         "original": {"volume_db": 0.0, "mute": False},
         "tracks": [{"id": "t1", "name": "Track 1", "volume_db": 0.0, "mute": False},
                    {"id": "t2", "name": "Track 2", "volume_db": 0.0, "mute": False}],
         "clips": [], "outputs": [], "master": dict(MASTER_DEFAULT)}
    os.makedirs(_root(mid), exist_ok=True)
    m["duration"] = get_from(json.loads(json.dumps(m)))["duration"]
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
        m = json.load(f)
    if "vclips" not in m:  # a mix from before video splicing: its one video becomes the video track
        spec = m.pop("video", None)
        m["vclips"] = []
        if spec:
            try:
                spec = norm_src(spec)
            except ValueError:
                spec = None
        if spec:
            m["vclips"] = [{**_vclip(spec, {"duration": m.get("duration") or 0})}]
        m["settings"] = dict(SETTINGS_DEFAULT)
        for k in ("fps", "width", "height", "has_audio"):
            m.pop(k, None)
    return m


MASTER_DEFAULT = {"target": "-14", "limiter": True, "ceiling": -1.0, "glue": False}
TARGETS = {"off": "Off", "-14": "Streaming (-14 LUFS)", "-16": "Web (-16 LUFS)", "-23": "Broadcast (-23 LUFS)"}
NR = {"off": 0, "light": 8, "medium": 14, "strong": 22}       # noise reduction, dB
LOWCUT = (0, 60, 80, 120, 200)                                  # Hz, 0 = off
HIGHCUT = (0, 12000, 8000, 5000, 3000)                          # Hz, 0 = off
CLIP_TARGET = -20.0                                              # match loudness to this (LUFS)


def get_from(m: dict) -> dict:
    """A stored mix with everything worked out: sources, output format, clip placement,
    duration, markers, and each sound clip's details."""
    m["master"] = {**MASTER_DEFAULT, **(m.get("master") or {})}
    m["settings"] = {**SETTINGS_DEFAULT, **(m.get("settings") or {})}
    srcs: dict[str, dict] = {}
    for v in m["vclips"]:
        k = v["skey"] = skey(v["src"])
        if k in srcs:
            continue
        try:
            s = source(v["src"])
            srcs[k] = {**{x: s[x] for x in ("name", "duration", "fps", "fps_frac", "width", "height", "sar", "has_audio",
                                            "markers")}, "ok": True}
        except (ValueError, LookupError, RuntimeError, OSError) as e:
            srcs[k] = {"ok": False, "error": str(e), "name": v["src"].get("file") or v["src"].get("id") or "missing",
                       "duration": v["out"], "has_audio": False, "markers": []}
    m["sources"] = srcs
    m["out"] = output_format(m, srcs)
    rows, m["duration"] = layout(m["vclips"], m["out"]["fps"])
    m["layout"], m["fps"] = rows, m["out"]["fps"]
    m["has_audio"] = any(srcs[v["skey"]]["has_audio"] and not v["mute"] for v in m["vclips"])
    m["video_ok"] = all(s["ok"] for s in srcs.values())
    bad = [s for s in srcs.values() if not s["ok"]]
    if bad:
        m["video_error"] = bad[0]["error"]
    markers = []
    for i, (v, r) in enumerate(zip(m["vclips"], rows)):
        s = srcs[v["skey"]]
        if i:
            markers.append({"t": round(r["start"], 3), "label": s["name"], "join": True})
        for mk in s["markers"]:  # the source's own joins and scene cuts, where they fall in this piece
            if v["in"] + 0.05 < mk["t"] < v["out"] - 0.05:
                markers.append({"t": round(r["start"] + mk["t"] - v["in"], 3), "label": mk["label"]})
    m["markers"] = markers
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


def get(mid: str) -> dict:
    return get_from(_read(mid))


def list_mixes() -> list[dict]:
    base = _root()
    out = []
    if os.path.isdir(base):
        for mid in sorted(os.listdir(base), reverse=True):
            if MID_RE.match(mid) and os.path.exists(_root(mid, "mix.json")):
                m = _read(mid)
                out.append({"id": mid, "name": m["name"], "updated": m["updated"], "clips": len(m["clips"]),
                            "videos": len(m["vclips"]), "duration": m.get("duration") or 0,
                            "outputs": len(m["outputs"])})
    return out


def _num(v, lo, hi, default=0.0) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return max(lo, min(hi, v))


def _save_vclips(items: list, old: list) -> list[dict]:
    durs = {}
    out = []
    for v in items[:MAX_VCLIPS]:
        try:
            spec = norm_src(v.get("src") or {})
        except (ValueError, LookupError):
            continue
        k = skey(spec)
        if k not in durs:
            try:
                durs[k] = source(spec)["duration"]
            except (ValueError, LookupError, RuntimeError, OSError):
                durs[k] = None  # missing: keep the clip as it was, so it comes back if the file does
        a = _num(v.get("in"), 0, 86400)
        b = _num(v.get("out"), 0, 86400, a + 1)
        if durs[k]:
            b = min(b, durs[k])
            a = min(a, max(0.0, b - 0.05))
        b = max(b, a + 0.05)
        tr = v.get("transition") or {}
        out.append({"id": str(v.get("id") or secrets.token_hex(4))[:20], "src": spec, "in": round(a, 4), "out": round(b, 4),
                    "transition": {"type": tr.get("type") if tr.get("type") in TRANSITIONS else "cut",
                                   "dur": round(_num(tr.get("dur"), 0.05, 5, 0.5), 3)},
                    "volume_db": _num(v.get("volume_db"), -60, 12), "mute": bool(v.get("mute"))})
    seen = set()
    for v in out:  # ids must be unique
        while v["id"] in seen:
            v["id"] = secrets.token_hex(4)
        seen.add(v["id"])
    return out


def save(mid: str, data: dict) -> dict:
    """Replace the editable parts (name, video clips, output settings, original audio, tracks,
    clips, master), validated."""
    m = _read(mid)
    if "name" in data:
        m["name"] = str(data["name"] or m["name"])[:120]
    if "vclips" in data:
        m["vclips"] = _save_vclips(data.get("vclips") or [], m["vclips"])
    st = {**SETTINGS_DEFAULT, **(m.get("settings") or {}), **(data.get("settings") or {})}
    size = str(st["size"])
    sm = re.match(r"^(\d{2,4})x(\d{2,4})$", size)
    if size != "auto" and not (sm and 64 <= int(sm[1]) <= 7680 and 64 <= int(sm[2]) <= 7680):
        size = "auto"
    if sm and size != "auto":
        size = f"{int(sm[1]) - int(sm[1]) % 2}x{int(sm[2]) - int(sm[2]) % 2}"
    m["settings"] = {"size": size, "fps": str(st["fps"]) if str(st["fps"]) in FPS_CHOICES else "auto",
                     "fit": st["fit"] if st["fit"] in FITS else "pad"}
    o = data.get("original") or m["original"]
    m["original"] = {"volume_db": _num(o.get("volume_db"), -60, 12), "mute": bool(o.get("mute"))}
    tracks = []
    for t in (data.get("tracks") or m["tracks"])[:32]:
        tid = str(t.get("id") or "")[:20]
        if tid and not any(x["id"] == tid for x in tracks):
            duck = str(t.get("duck_source") or "")[:20]
            tracks.append({"id": tid, "name": str(t.get("name") or tid)[:60], "volume_db": _num(t.get("volume_db"), -60, 12),
                           "mute": bool(t.get("mute")), "duck_source": duck if duck != tid else "",
                           "duck_db": _num(t.get("duck_db"), 0, 40, 10)})
    if not tracks:
        raise ValueError("a mix needs at least one track")
    ids = {t["id"] for t in tracks}
    for t in tracks:  # a duck source must still exist
        if t["duck_source"] not in ("", "original") and t["duck_source"] not in ids:
            t["duck_source"] = ""
    mst = {**MASTER_DEFAULT, **(m.get("master") or {}), **(data.get("master") or {})}
    m["master"] = {"target": mst["target"] if str(mst["target"]) in TARGETS else "off",
                   "limiter": bool(mst["limiter"]), "ceiling": _num(mst["ceiling"], -6, 0, -1.0), "glue": bool(mst["glue"])}
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
                      "fade_in": _num(c.get("fade_in"), 0, length / 2), "fade_out": _num(c.get("fade_out"), 0, length / 2),
                      "match": bool(c.get("match")), "nr": c.get("nr") if c.get("nr") in NR else "off",
                      "lowcut": int(c.get("lowcut") or 0) if int(c.get("lowcut") or 0) in LOWCUT else 0,
                      "highcut": int(c.get("highcut") or 0) if int(c.get("highcut") or 0) in HIGHCUT else 0,
                      "declick": bool(c.get("declick")), "pan": _num(c.get("pan"), -1, 1)})
    m["tracks"], m["clips"], m["updated"] = tracks, clips, time.time()
    full = get_from(json.loads(json.dumps(m)))
    m["duration"] = full["duration"]
    _write(m)
    return get(mid)


def delete(mid: str) -> None:
    _read(mid)
    shutil.rmtree(_root(mid), ignore_errors=True)


def source_path(mid: str, key: str) -> str:
    """The file behind one of this mix's video sources (only sources the mix uses)."""
    for v in _read(mid)["vclips"]:
        if skey(v["src"]) == key:
            return resolve_video(v["src"])["path"]
    raise LookupError("that video isn't in this mix")


def thumb(mid: str, key: str, t: float) -> str:
    """A cached 160 px frame of a source at time t (for the video track)."""
    path = source_path(mid, key)
    st = os.stat(path)
    h = hashlib.sha1(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:14]
    p = _root("_thumbs", f"{h}-{int(round(max(0.0, t) * 1000)):09d}.jpg")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{os.getpid()}-{threading.get_ident()}.part.jpg"
        media.grab_frame(path, max(0.0, t), 160).save(tmp, quality=80)
        os.replace(tmp, p)
    return p


# ==== the video browser ===================================================================================

def browse() -> list[dict]:
    """Every video that can go on the video track, with what the browser shows and sorts by:
    Colour match renders (the joined chain and any per-segment clips) and library videos."""
    items = []
    for x in colormatch.list_sessions():
        if not x.get("rendered"):
            continue
        try:
            s = colormatch.load(x["id"])
        except (ValueError, LookupError):
            continue
        out = s.get("outputs") or {}
        when = out.get("rendered") or s.get("created") or 0
        folder = "Colour match " + time.strftime("%Y-%m-%d %H:%M", time.localtime(s.get("created") or when))
        names = [sg["name"] for sg in s["segments"]]
        files = [(out.get("joined"), f"{len(names)} segments: " + ", ".join(names))]
        files += [(c, f"segment {k + 1} of {len(names)}: {names[k] if k < len(names) else c}")
                  for k, c in enumerate(out.get("clips") or [])]
        for f, label in files:
            if not f:
                continue
            spec = {"kind": "colormatch", "session": s["id"], "file": f}
            try:
                src = source(spec)
            except (ValueError, LookupError, RuntimeError, OSError):
                continue
            items.append({"key": skey(spec), "spec": spec, "chain": True, "name": label, "file": f, "folder": folder,
                          "mtime": when, "duration": src["duration"], "width": src["width"], "height": src["height"],
                          "fps": src["fps"], "has_audio": src["has_audio"],
                          "url": f"/api/colormatch/{s['id']}/file/{f}"})
    for v in library.list_videos():
        info = v["info"] or {}
        w, h = info.get("width") or 0, info.get("height") or 0
        if info.get("rotation") in (90, 270):
            w, h = h, w
        sub = os.path.dirname(v.get("rel") or "").replace("\\", "/")
        folder = os.path.basename(os.path.normpath(v.get("root") or "")) + (f"/{sub}" if sub else "")
        spec = {"kind": "library", "id": v["id"]}
        items.append({"key": skey(spec), "spec": spec, "chain": False, "name": v["name"], "file": v["name"],
                      "folder": folder, "mtime": v.get("mtime") or 0, "duration": info.get("duration") or 0,
                      "width": w, "height": h, "fps": info.get("fps") or 0, "has_audio": bool(info.get("has_audio")),
                      "url": f"/media/video/{v['id']}"})
    return items


_poster_sem = threading.BoundedSemaphore(3)   # a grid asks for many at once; keep the machine responsive


def poster(spec: dict) -> str:
    """A cached filmstrip of a video's first, middle and last frames, side by side (240 px each)."""
    spec = norm_src(spec, fill=True)
    path = resolve_video(spec)["path"]
    st = os.stat(path)
    h = hashlib.sha1(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16]
    p = _root("_posters", f"{h}.jpg")
    if os.path.exists(p):
        return p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with _poster_sem:
        if os.path.exists(p):
            return p
        tmp = f"{p}.{os.getpid()}-{threading.get_ident()}.part.jpg"
        try:
            dur = source(spec)["duration"]
            fit = "scale=240:-2,setsar=1"
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ss", f"{dur / 2:.3f}", "-i", path,
                                "-sseof", "-0.25", "-i", path, "-filter_complex",
                                f"[0:v]{fit}[a];[1:v]{fit}[b];[2:v]{fit}[c];[a][b][c]hstack=3",
                                "-frames:v", "1", "-q:v", "4", tmp], capture_output=True, **_NOWIN)
            if r.returncode != 0 or not os.path.exists(tmp):
                # very short or unusual files: grab the three frames one by one
                from PIL import Image
                ims = [media.grab_frame(path, t, 240) for t in (0.0, dur / 2, max(0.0, dur - 0.1))]
                strip = Image.new("RGB", (240 * 3, max(i.height for i in ims)))
                for k, im in enumerate(ims):
                    strip.paste(im, (240 * k, 0))
                strip.save(tmp, quality=82)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return p


def output_path(mid: str, name: str) -> str:
    _read(mid)
    if not re.match(r"^[^/\\]{1,200}$", name or "") or name in (".", "..") or name == "mix.json":
        raise ValueError("bad file name")
    return _root(mid, name)


# ==== per-clip cleanup (cached) ========================================================================

_fx_locks: dict[str, threading.Lock] = {}
_fx_guard = threading.Lock()
_NOWIN = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


def clip_fx(c: dict) -> dict | None:
    """The processing a clip asks for, in canonical form; None when it needs none."""
    fx = {"match": bool(c.get("match")), "nr": c.get("nr") or "off", "lowcut": int(c.get("lowcut") or 0),
          "highcut": int(c.get("highcut") or 0), "declick": bool(c.get("declick"))}
    if not fx["match"] and fx["nr"] == "off" and not fx["lowcut"] and not fx["highcut"] and not fx["declick"]:
        return None
    return fx


def fx_key(fx: dict) -> str:
    return f"m{int(fx['match'])}-n{fx['nr']}-l{fx['lowcut']}-h{fx['highcut']}-d{int(fx['declick'])}"


def parse_fx(key: str) -> dict:
    m = re.match(r"^m([01])-n(off|light|medium|strong)-l(\d+)-h(\d+)-d([01])$", key or "")
    if not m or int(m[3]) not in LOWCUT or int(m[4]) not in HIGHCUT:
        raise ValueError("bad processing settings")
    return {"match": m[1] == "1", "nr": m[2], "lowcut": int(m[3]), "highcut": int(m[4]), "declick": m[5] == "1"}


def _noise_floor(path: str) -> float:
    """The clip's noise floor (dBFS): the level of its quietest 10%, clamped to what afftdn accepts.
    afftdn needs this set; its own noise tracking never catches up on short clips."""
    lv = sounds._features(path)["level"]
    lv = lv[lv > -100]
    return float(max(-80.0, min(-20.0, np.percentile(lv, 10)))) if lv.size else -50.0


def _fx_filters(fx: dict, path: str) -> list[str]:
    f = []
    if fx["declick"]:
        f.append("adeclick")
    if fx["lowcut"]:
        f.append(f"highpass=f={fx['lowcut']}:poles=2")
    if fx["highcut"]:
        f.append(f"lowpass=f={fx['highcut']}:poles=2")
    if NR.get(fx["nr"]):
        f.append(f"afftdn=nr={NR[fx['nr']]}:nf={_noise_floor(path):.1f}")
    return f


def loudness(path: str) -> dict:
    """Integrated loudness (LUFS) and peak (dBFS). Clips shorter than the 400 ms loudness
    window have no integrated value, so their RMS level stands in for it."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-i", path, "-af",
                          "ebur128=peak=sample,astats=metadata=0:measure_perchannel=none", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", **_NOWIN).stderr
    i = re.findall(r"I:\s+(-?[\d.]+) LUFS", out)
    pk = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", out)
    rms = re.findall(r"RMS level dB:\s*(-?[\d.]+|-inf)", out)
    lufs = float(i[-1]) if i else None
    if lufs is None or lufs <= -69:
        lufs = float(rms[-1]) if rms and rms[-1] != "-inf" else None
    return {"lufs": lufs, "peak": float(pk[-1]) if pk and pk[-1] != "-inf" else None}


def processed_path(sound_id: int, fx: dict) -> str:
    """The sound with its cleanup applied, as 48 kHz stereo float WAV (no clipping, whatever the
    gain), built once per sound version and settings, safely under concurrent requests."""
    snd = sounds.get(sound_id)
    st = os.stat(snd["path"])
    h = hashlib.sha1(f"{sound_id}:{st.st_size}:{st.st_mtime_ns}:{fx_key(fx)}".encode()).hexdigest()[:20]
    out = _root("_fx", f"{h}.wav")
    if os.path.exists(out):
        return out
    with _fx_guard:
        lock = _fx_locks.setdefault(out, threading.Lock())
    with lock:
        if os.path.exists(out):
            return out
        os.makedirs(_root("_fx"), exist_ok=True)
        tmp = f"{out}.{os.getpid()}-{threading.get_ident()}.part.wav"
        try:
            filters = ["aresample=48000", "aformat=channel_layouts=stereo"] + _fx_filters(fx, snd["path"])
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", snd["path"], "-af", ",".join(filters),
                            "-c:a", "pcm_f32le", tmp], check=True, capture_output=True, **_NOWIN)
            if fx["match"]:
                lv = loudness(tmp)["lufs"]
                if lv is not None:
                    gain = max(-30.0, min(30.0, CLIP_TARGET - lv))
                    tmp2 = tmp + ".g.wav"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-af", f"volume={gain:.2f}dB",
                                    "-c:a", "pcm_f32le", tmp2], check=True, capture_output=True, **_NOWIN)
                    os.replace(tmp2, tmp)
            os.replace(tmp, out)
        finally:
            for p in (tmp, tmp + ".g.wav"):
                if os.path.exists(p):
                    os.remove(p)
    return out


def clip_source(c: dict) -> str | None:
    """File a clip plays from: the processed copy when it has cleanup, else the sound itself."""
    try:
        fx = clip_fx(c)
        return processed_path(c["sound"], fx) if fx else sounds.get(c["sound"])["path"]
    except (LookupError, OSError, subprocess.CalledProcessError):
        return None


# ==== ducking =============================================================================================

ATTACK, RELEASE = 0.15, 0.5   # seconds to dip before a sound starts, and to recover after it ends


def _merge(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    out = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _video_activity(m: dict) -> list[tuple[float, float]]:
    """Where the video's own audio is audible (its loud parts, not room tone)."""
    path = track_audio(None, m) if m.get("has_audio") else None
    if not path:
        return []
    st = os.stat(path)
    cache = _root("_fx", "act-" + hashlib.sha1(f"{path}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16] + ".json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return [tuple(x) for x in json.load(f)]
    feats = sounds._features(path)
    lv = np.convolve(feats["level"].astype(np.float64), np.ones(15) / 15, mode="same")  # 150 ms smoothing
    loud = float(np.percentile(lv[lv > -100], 95)) if (lv > -100).any() else -100.0
    hop = sounds.HOP / sounds.SR
    spans = [(a * hop, b * hop) for a, b in sounds._runs(lv > max(loud - 25, -55))]
    spans = [x for x in _merge(spans, 0.3) if x[1] - x[0] >= 0.1]
    os.makedirs(_root("_fx"), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(spans, f)
    return spans


def duck_curve(m: dict, t: dict) -> list[list[float]]:
    """Gain breakpoints [[time, gain], ...] for a ducked track (empty: no ducking)."""
    src = t.get("duck_source") or ""
    if not src or (t.get("duck_db") or 0) <= 0:
        return []
    if src == "original":
        if m["original"]["mute"]:
            return []
        spans = _video_activity(m)
    else:
        other = next((x for x in m["tracks"] if x["id"] == src), None)
        if not other or other["mute"]:
            return []
        spans = [(c["start"], c["start"] + c["length"]) for c in m["clips"] if c["track"] == src]
    spans = _merge(spans, ATTACK + RELEASE)
    if not spans:
        return []
    g, d = 10 ** (-t["duck_db"] / 20), m["duration"]
    pts = [[0.0, 1.0]]
    for a, b in spans:
        pts += [[max(0.0, a - ATTACK), 1.0], [max(0.0, a), g], [min(d, b), g], [min(d, b + RELEASE), 1.0]]
    pts.append([d, 1.0])
    out = []  # tidy: keep times increasing
    for p in pts:
        if out and p[0] <= out[-1][0]:
            out[-1] = [out[-1][0], min(out[-1][1], p[1])]
        else:
            out.append([round(p[0], 4), round(p[1], 5)])
    return out


def duck_curves(mid: str) -> dict:
    m = get(mid)
    return {t["id"]: duck_curve(m, t) for t in m["tracks"]}


# ==== the video track: its audio and its picture ==============================================================

_trk_locks: dict[str, threading.Lock] = {}


def _clip_paths(m: dict) -> list[str]:
    bad = [s for s in m["sources"].values() if not s["ok"]]
    if bad:
        raise LookupError(f"a video in this mix is missing: {bad[0]['error']}")
    return [resolve_video(v["src"])["path"] for v in m["vclips"]]


def track_audio(job, m: dict) -> str | None:
    """The video track's own audio, spliced exactly as the picture is (trims, per-clip volume
    and mute, crossfades at transitions), as a cached 48 kHz stereo float WAV. None when the
    track has no audio to play."""
    if not m["vclips"] or not m["has_audio"]:
        return None
    paths = _clip_paths(m)
    F, rows = m["out"]["fps"], m["layout"]
    parts = []
    for v, r, p in zip(m["vclips"], rows, paths):
        st = os.stat(p)
        parts.append([p, st.st_size, st.st_mtime_ns, v["in"], r["frames"], r["dframes"], v["volume_db"],
                      v["mute"] or not m["sources"][v["skey"]]["has_audio"]])
    h = hashlib.sha1(json.dumps([F, parts]).encode()).hexdigest()[:20]
    out = _root("_fx", f"trk-{h}.wav")
    if os.path.exists(out):
        return out
    with _fx_guard:
        lock = _trk_locks.setdefault(out, threading.Lock())
    with lock:
        if os.path.exists(out):
            return out
        os.makedirs(_root("_fx"), exist_ok=True)
        inputs, chains, k = [], [], 0
        for i, (p, _, _, a, n, _, vol, silent) in enumerate(parts):
            L = n / F
            if silent:
                chains.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={L:.5f},"
                              f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{i}]")
                continue
            inputs += ["-ss", f"{a:.4f}", "-t", f"{L + 0.5:.4f}", "-i", p]
            chains.append(f"[{k}:a:0]aresample=48000,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,"
                          f"asetpts=PTS-STARTPTS,volume={_db(vol)},apad=whole_dur={L:.5f},atrim=duration={L:.5f}[a{i}]")
            k += 1
        acc = "[a0]"
        for i in range(1, len(parts)):
            d = rows[i]["dframes"] / F
            if d > 0:  # transitions blend the sound too
                chains.append(f"{acc}[a{i}]acrossfade=d={d:.5f}:c1=tri:c2=tri[j{i}]")
            else:
                chains.append(f"{acc}[a{i}]concat=n=2:v=0:a=1[j{i}]")
            acc = f"[j{i}]"
        tmpd = tempfile.mkdtemp(dir=_root("_fx"))
        try:
            g = os.path.join(tmpd, "graph.txt")
            with open(g, "w", encoding="utf-8") as f:
                f.write(";\n".join(chains))
            tmp = os.path.join(tmpd, "track.wav")
            if job:
                job.update(message="splicing the video's audio", progress=0, total=int(m["duration"]))
            media.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-/filter_complex", g, "-map", acc,
                       "-c:a", "pcm_f32le", tmp], job=job, duration=m["duration"])
            os.replace(tmp, out)
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)
    return out


def _copyable(m: dict) -> bool:
    """One whole clip at its own size and rate: the picture can be copied, not re-encoded."""
    if len(m["vclips"]) != 1:
        return False
    v, s, o = m["vclips"][0], m["sources"][m["vclips"][0]["skey"]], m["out"]
    return (v["in"] < 0.5 / o["fps"] and abs(v["out"] - s["duration"]) < 1 / o["fps"]
            and s["fps_frac"] == o["fps_frac"] and (s["width"], s["height"]) == (o["width"], o["height"]))


def _fit(i: int, fit: str, W: int, H: int, tail: str) -> str:
    big = f"scale={W}:{H}:force_original_aspect_ratio=increase:force_divisible_by=2,crop={W}:{H}"
    small = f"scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    if fit == "crop":
        return f"{big},{tail}"
    if fit == "blur":
        return (f"split[bg{i}][fg{i}];[bg{i}]{big},gblur=sigma=24[bb{i}];[fg{i}]{small}[ff{i}];"
                f"[bb{i}][ff{i}]overlay=(W-w)/2:(H-h)/2,{tail}")
    return f"{small},pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,{tail}"


def track_video(job, m: dict, tmp: str) -> tuple[str, bool]:
    """(the video track's picture, copied?). Each clip is cut exactly, fitted to the output
    size, put on the output frame grid, and joined by its transition; encoded once."""
    paths = _clip_paths(m)
    if _copyable(m):
        return paths[0], True
    o, rows = m["out"], m["layout"]
    F, W, H = o["fps"], o["width"], o["height"]
    inputs, chains = [], []
    for i, (v, r, p) in enumerate(zip(m["vclips"], rows, paths)):
        s = m["sources"][v["skey"]]
        inputs += ["-ss", f"{v['in']:.4f}", "-t", f"{r['len'] + 1:.4f}", "-i", p]
        pre = f"[{i}:v:0]setpts=PTS-STARTPTS"
        if abs(s["sar"] - 1) > 0.01:  # non-square pixels: make them square first
            pre += ",scale=trunc(iw*sar/2)*2:ih,setsar=1"
        tail = (f"setsar=1,fps={o['fps_frac']},tpad=stop_mode=clone:stop_duration=2,trim=end_frame={r['frames']},"
                f"setpts=PTS-STARTPTS,format=yuv420p,settb=AVTB[v{i}]")
        chains.append(f"{pre},{_fit(i, o['fit'], W, H, tail)}")
    acc = "[v0]"
    for i in range(1, len(rows)):
        r, v = rows[i], m["vclips"][i]
        if r["dframes"]:
            kind = "fadeblack" if v["transition"]["type"] == "fadeblack" else "fade"
            chains.append(f"{acc}[v{i}]xfade=transition={kind}:duration={r['dframes'] / F:.5f}:offset={r['sframe'] / F:.5f}[x{i}]")
        else:
            chains.append(f"{acc}[v{i}]concat=n=2:v=1:a=0[x{i}]")
        acc = f"[x{i}]"
    g = os.path.join(tmp, "video-graph.txt")
    with open(g, "w", encoding="utf-8") as f:
        f.write(";\n".join(chains))
    out = os.path.join(tmp, "video.mp4")
    if job:
        job.update(message=f"splicing {len(rows)} video clip{'s' if len(rows) != 1 else ''}", progress=0,
                   total=int(m["duration"]))
    media.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-/filter_complex", g, "-map", acc, "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-r", o["fps_frac"], "-frames:v", str(rows[-1]["sframe"] + rows[-1]["frames"]), out],
              job=job, duration=m["duration"])
    return out, False


# ==== rendering ==========================================================================================

def _db(x: float) -> str:
    return f"{x:.2f}dB"


def _pan(p: float) -> str:
    """Web Audio's StereoPannerNode law for stereo input, so the render pans like the preview."""
    if abs(p) < 1e-3:
        return ""
    if p <= 0:
        x = (p + 1) * math.pi / 2
        return f"pan=stereo|c0=c0+{math.cos(x):.5f}*c1|c1={math.sin(x):.5f}*c1"
    x = p * math.pi / 2
    return f"pan=stereo|c0={math.cos(x):.5f}*c0|c1=c1+{math.sin(x):.5f}*c0"


def build_graph(m: dict, sources: dict[str, str], envs: dict[str, str]) -> tuple[list[str], str, int]:
    """(extra ffmpeg inputs, filter graph ending in [pre], layers). Input 0 is the video track's audio.
    sources: clip id -> file to play; envs: track id -> ducking envelope file."""
    tracks = {t["id"]: t for t in m["tracks"]}
    d = m["duration"]
    inputs, chains, finals = [], [], []
    if m.get("has_audio") and not m["original"]["mute"]:
        chains.append(f"[0:a:0]aresample=48000,aformat=channel_layouts=stereo,volume={_db(m['original']['volume_db'])},"
                      f"apad=whole_dur={d:.4f},atrim=duration={d:.4f}[orig]")
        finals.append("[orig]")
    k = 1
    per_track: dict[str, list[str]] = {}
    for c in m["clips"]:
        t = tracks.get(c["track"])
        if not t or t["mute"] or c["id"] not in sources or c["start"] >= d:
            continue
        length = min(c["length"], d - c["start"])
        inputs += (["-stream_loop", "-1"] if c["loop"] else []) + ["-i", sources[c["id"]]]
        f = [f"[{k}:a:0]aresample=48000", "aformat=channel_layouts=stereo",
             f"atrim=start={c['offset']:.4f}:duration={length:.4f}", "asetpts=PTS-STARTPTS"]
        if c["fade_in"] > 0:
            f.append(f"afade=t=in:st=0:d={c['fade_in']:.4f}")
        if c["fade_out"] > 0:
            f.append(f"afade=t=out:st={max(0.0, length - c['fade_out']):.4f}:d={c['fade_out']:.4f}")
        f.append(f"volume={_db(c['volume_db'])}")
        if _pan(c.get("pan") or 0):
            f.append(_pan(c["pan"]))
        f.append(f"adelay=delays={int(round(c['start'] * 1000))}:all=1")
        chains.append(",".join(f) + f"[c{k}]")
        per_track.setdefault(t["id"], []).append(f"[c{k}]")
        k += 1
    n_layers = len(finals) + sum(len(v) for v in per_track.values())
    for tid, labels in per_track.items():
        t = tracks[tid]
        chains.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0,"
                      f"volume={_db(t['volume_db'])},apad=whole_dur={d:.4f},atrim=duration={d:.4f}[t_{tid}]")
        if tid in envs:  # ducking: multiply by the gain curve
            inputs += ["-i", envs[tid]]
            chains.append(f"[{k}:a:0]aresample=48000,aformat=channel_layouts=stereo,apad=whole_dur={d:.4f},"
                          f"atrim=duration={d:.4f}[e_{tid}]")
            chains.append(f"[t_{tid}][e_{tid}]amultiply[td_{tid}]")
            finals.append(f"[td_{tid}]")
            k += 1
        else:
            finals.append(f"[t_{tid}]")
    if not finals:
        raise ValueError("nothing to mix: add sounds, or unmute a track or the original audio")
    glue = ",acompressor=threshold=-18dB:ratio=2:attack=20:release=250:makeup=1" if m["master"]["glue"] else ""
    chains.append(f"{''.join(finals)}amix=inputs={len(finals)}:duration=longest:normalize=0,"
                  f"apad=whole_dur={d:.4f},atrim=duration={d:.4f}{glue}[pre]")
    return inputs, ";\n".join(chains), n_layers


def _write_env(pts: list[list[float]], d: float, path: str) -> None:
    """A gain curve as a 1 kHz mono float WAV (resampled to 48 kHz inside the graph)."""
    t = np.arange(0, d + 0.01, 0.001)
    g = np.interp(t, [p[0] for p in pts], [p[1] for p in pts]).astype(np.float32)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", "1000", "-ac", "1", "-i", "-",
                    "-c:a", "pcm_f32le", path], input=g.tobytes(), check=True, capture_output=True, **_NOWIN)


def _mix_audio(job, m: dict, tmp: str) -> tuple[str, int]:
    """Stage one: every layer mixed (with cleanup, pan, ducking, glue) into a float WAV."""
    if m["duration"] <= 0:
        raise ValueError("add a video to the mix first")
    trk = track_audio(job, m)
    src = ["-i", trk] if trk else ["-f", "lavfi", "-t", f"{m['duration']:.4f}", "-i", "anullsrc=r=48000:cl=stereo"]
    sources = {}
    todo = [c for c in m["clips"]]
    for n, c in enumerate(todo):
        if job:
            job.check()
            job.update(message=f"preparing sounds {n + 1}/{len(todo)}", progress=n, total=len(todo))
        p = clip_source(c)
        if p:
            sources[c["id"]] = p
    envs = {}
    for t in m["tracks"]:
        pts = duck_curve(m, t)
        if pts:
            envs[t["id"]] = os.path.join(tmp, f"env_{t['id']}.wav")
            _write_env(pts, m["duration"], envs[t["id"]])
    inputs, graph, n = build_graph(m, sources, envs)
    gpath = os.path.join(tmp, "graph.txt")
    with open(gpath, "w", encoding="utf-8") as f:
        f.write(graph)
    out = os.path.join(tmp, "mix.wav")
    if job:
        job.update(message=f"mixing {n} layer{'s' if n != 1 else ''}", progress=0, total=int(m["duration"]))
    media.run(["ffmpeg", "-y", "-loglevel", "error", *src, *inputs, "-/filter_complex", gpath,
               "-map", "[pre]", "-c:a", "pcm_f32le", out], job=job, duration=m["duration"])
    return out, n


def _loudnorm_first_pass(path: str, target: float, ceiling: float) -> dict:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-i", path, "-af",
                          f"loudnorm=I={target}:TP={ceiling}:LRA=11:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", **_NOWIN).stderr
    j = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", out, re.S)
    if not j:
        raise RuntimeError("couldn't measure the mix loudness")
    return json.loads(j.group(0))


def _master_chain(m: dict, meas: dict | None, aim: float | None) -> str:
    ms = m["master"]
    f = []
    if meas is not None and aim is not None:
        f.append(f"loudnorm=I={aim:.2f}:TP={ms['ceiling']}:LRA=11:linear=true:"
                 f"measured_I={meas['input_i']}:measured_TP={meas['input_tp']}:measured_LRA={meas['input_lra']}:"
                 f"measured_thresh={meas['input_thresh']}:offset={meas['target_offset']},aresample=48000")
    if ms["limiter"]:
        f.append(f"alimiter=limit={10 ** (ms['ceiling'] / 20):.4f}:level=false:attack=2:release=50")
    return ",".join(f) or "anull"


def _master_filters(m: dict, mix_wav: str) -> str:
    """Loudness target and limiter. The limiter trims peaks after the loudness step, which
    lowers the result a little, so an audio-only trial pass measures what the chain really
    gives and the aim is nudged to land on the target."""
    ms = m["master"]
    if ms["target"] == "off":
        return _master_chain(m, None, None)
    target = float(ms["target"])
    meas = _loudnorm_first_pass(mix_wav, target, ms["ceiling"])
    if float(meas["input_i"]) <= -70:  # silence can't be normalised
        return _master_chain(m, None, None)
    aim = target
    for _ in range(2):
        chain = _master_chain(m, meas, aim)
        trial = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-i", mix_wav, "-af",
                                chain + ",ebur128", "-f", "null", "-"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", **_NOWIN).stderr
        got = re.findall(r"I:\s+(-?[\d.]+) LUFS", trial)
        if not got or abs(float(got[-1]) - target) <= 0.3:
            break
        aim = max(-40.0, min(-5.0, aim + max(-3.0, min(3.0, target - float(got[-1])))))
        meas = _loudnorm_first_pass(mix_wav, aim, ms["ceiling"])
    return _master_chain(m, meas, aim)


def measure(job, mid: str) -> dict:
    """Mix the audio and measure it, so the preview can play at the level the render will have."""
    m = get(mid)
    with tempfile.TemporaryDirectory() as tmp:
        wav, _ = _mix_audio(job, m, tmp)
        lv = loudness(wav)
    m = _read(mid)
    m["measured"] = {"lufs": lv["lufs"], "peak": lv["peak"], "at": time.time()}
    _write(m)
    return m["measured"]


def render(job, mid: str, to_library: bool = False) -> dict:
    m = get(mid)
    if not m["vclips"]:
        raise ValueError("add a video to the mix first")
    with tempfile.TemporaryDirectory() as tmp:
        src, copied = track_video(job, m, tmp)
        ext = os.path.splitext(src)[1].lower() if copied else ".mp4"
        ext = ext if ext in (".mp4", ".mov", ".mkv", ".m4v") else ".mp4"
        safe = re.sub(r"[^\w\- .()]+", "_", m["name"]).strip(" ._") or "mix"
        from ..fileops import _free
        out = _free(_root(mid, f"{safe}-mixed{ext}"))
        wav, n = _mix_audio(job, m, tmp)
        if job:
            job.update(message="levels (loudness target, limiter)", progress=0, total=int(m["duration"]))
        af = _master_filters(m, wav)

        def mux(chain: str) -> None:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
                   "-af", chain, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-cutoff", "20000",
                   "-t", f"{m['duration']:.4f}"]
            if ext in (".mp4", ".mov", ".m4v"):
                cmd += ["-movflags", "+faststart"]
            media.run(cmd + [out], job=job, duration=m["duration"])

        mux(af)
        final = loudness(out)
        ms = m["master"]
        if ms["target"] != "off" and final["lufs"] is not None and abs(final["lufs"] - float(ms["target"])) > 0.3:
            # AAC encoding shifts loudness slightly (more on noisy, bright material): correct and re-encode
            fix = max(-3.0, min(3.0, float(ms["target"]) - final["lufs"]))
            lim = f",alimiter=limit={10 ** (ms['ceiling'] / 20):.4f}:level=false:attack=2:release=50" if ms["limiter"] else ""
            if job:
                job.update(message=f"fine-tuning the level by {fix:+.1f} dB")
            mux(f"{af},volume={fix:.2f}dB{lim}")
            final = loudness(out)
    o = m["out"]
    m = _read(mid)
    entry = {"file": os.path.basename(out), "rendered": time.time(), "layers": n, "videos": len(m["vclips"]),
             "reencoded": not copied, "size": f"{o['width']}x{o['height']}", "fps": o["fps_frac"],
             "lufs": final["lufs"], "peak": final["peak"]}
    if to_library:  # the mixes folder acts as its root, like uploads do
        try:
            entry["library_id"] = library.add_path(out, root=_root())["id"]
        except (OSError, ValueError):
            entry["library_id"] = None
    m["outputs"] = [entry] + m["outputs"][:9]
    _write(m)
    return {"mix": mid, "file": entry["file"], "layers": n, "lufs": final["lufs"], "peak": final["peak"]}
