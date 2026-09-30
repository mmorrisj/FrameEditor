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
         "clips": [], "outputs": [], "master": dict(MASTER_DEFAULT)}
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


MASTER_DEFAULT = {"target": "-14", "limiter": True, "ceiling": -1.0, "glue": False}
TARGETS = {"off": "Off", "-14": "Streaming (-14 LUFS)", "-16": "Web (-16 LUFS)", "-23": "Broadcast (-23 LUFS)"}
NR = {"off": 0, "light": 8, "medium": 14, "strong": 22}       # noise reduction, dB
LOWCUT = (0, 60, 80, 120, 200)                                  # Hz, 0 = off
HIGHCUT = (0, 12000, 8000, 5000, 3000)                          # Hz, 0 = off
CLIP_TARGET = -20.0                                              # match loudness to this (LUFS)


def get(mid: str) -> dict:
    """The mix, with its video's markers and each clip's sound details for the editor."""
    m = _read(mid)
    m["master"] = {**MASTER_DEFAULT, **(m.get("master") or {})}
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
    if not m.get("has_audio"):
        return []
    path = resolve_video(m["video"])["path"]
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
    """(extra ffmpeg inputs, filter graph ending in [pre], layers). Input 0 is the video.
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
    src = resolve_video(m["video"])["path"]
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
    media.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, *inputs, "-/filter_complex", gpath,
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
    src = resolve_video(m["video"])["path"]
    ext = os.path.splitext(src)[1].lower()
    ext = ext if ext in (".mp4", ".mov", ".mkv", ".m4v") else ".mp4"
    safe = re.sub(r"[^\w\- .()]+", "_", m["name"]).strip(" ._") or "mix"
    from ..fileops import _free
    out = _free(_root(mid, f"{safe}-mixed{ext}"))
    with tempfile.TemporaryDirectory() as tmp:
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
    m = _read(mid)
    entry = {"file": os.path.basename(out), "rendered": time.time(), "layers": n,
             "lufs": final["lufs"], "peak": final["peak"]}
    if to_library:  # the mixes folder acts as its root, like uploads do
        try:
            entry["library_id"] = library.add_path(out, root=_root())["id"]
        except (OSError, ValueError):
            entry["library_id"] = None
    m["outputs"] = [entry] + m["outputs"][:9]
    _write(m)
    return {"mix": mid, "file": entry["file"], "layers": n, "lufs": final["lufs"], "peak": final["peak"]}
