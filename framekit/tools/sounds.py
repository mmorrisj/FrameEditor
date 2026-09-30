"""Sounds: cut audio into clips and keep them in a sound library for layering.

  sources   any library video with audio, or an audio file you upload
            (WAV, MP3, FLAC, OGG, M4A, AIFF, OPUS). Nothing is extracted first:
            the cutter reads the audio straight from the source.
  analysis  computed once per source and cached, every 10 ms: level (dBFS),
            waveform peaks, and onset strength (spectral flux, where sounds
            start). Gaps are spans quieter than a threshold set from the
            file's own loud level, so room tone and digital silence both work.
            Sounds are what lies between gaps; onsets mark hits inside them.
  library   clips are saved as lossless 48 kHz / 24-bit WAV files under the
            sound folder, in one subfolder per category, so they are usable
            outside FrameKit. Name, category, tags, length, level, a loopable
            flag, favourites and where each clip came from (source and time
            range) are kept in the database. Files dropped into the folder by
            hand are picked up by a rescan (their category is their subfolder).

Categories: a small preset list, replaceable in framekit.json:
    "sounds": {"categories": ["ambience", "foley", ...]}
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time

import numpy as np

from .. import config, library
from ..db import cache_get, cache_put, db

DEFAULT_CATEGORIES = ["ambience", "foley", "footsteps", "impacts", "voice", "music", "other"]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff", ".opus", ".wma"}
SR = 24000                 # analysis sample rate
HOP = 240                  # 10 ms
WIN = 512                  # onset-strength window
OUT_RATE = 48000           # saved clips
_NOWIN = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
_KEY = re.compile(r"^(v:[0-9a-f]{12}|a:[^/\\:*?\"<>|]{1,200})$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, name TEXT, category TEXT, tags TEXT,
    duration REAL, level REAL, peak REAL, channels INTEGER, loop INTEGER DEFAULT 0, fav INTEGER DEFAULT 0,
    source TEXT, source_name TEXT, source_start REAL, source_end REAL, notes TEXT, envelope TEXT,
    size INTEGER, mtime REAL, created REAL
);
"""
_schema_ok: set[str] = set()


def _db():
    if config.DB_PATH not in _schema_ok:
        with db() as c:
            c.executescript(SCHEMA)
        _schema_ok.add(config.DB_PATH)
    return db()


def sounds_dir() -> str:
    return config.env_path("FRAMEKIT_SOUNDS", os.path.join(config.WORK, "sounds"))


def sources_dir() -> str:
    return os.path.join(config.WORK, "audio-sources")


def _cache_dir() -> str:
    return os.path.join(config.WORK, "sound-cache")


def categories() -> list[str]:
    cats = config.settings("sounds").get("categories") or DEFAULT_CATEGORIES
    seen = [str(c).strip() for c in cats if str(c).strip()]
    with _db() as c:  # categories already used by saved sounds stay visible
        for r in c.execute("SELECT DISTINCT category FROM sounds WHERE category IS NOT NULL"):
            if r["category"] and r["category"] not in seen:
                seen.append(r["category"])
    return seen


def _safe(name: str) -> str:
    s = re.sub(r"[^\w\- .()]+", "_", name, flags=re.UNICODE).strip(" ._")
    return s[:120] or "sound"


# ==== sources =======================================================================================

def source_path(key: str) -> tuple[str, str]:
    """(file path, display name) for a source key: v:<video id> or a:<uploaded file name>."""
    if not _KEY.match(key or ""):
        raise ValueError("bad source")
    if key.startswith("v:"):
        v = library.require(key[2:])
        if not (v.get("info") or {}).get("has_audio"):
            raise ValueError("that video has no audio")
        return v["path"], v["name"]
    p = os.path.join(sources_dir(), key[2:])
    if not os.path.isfile(p) or not config.inside(p, sources_dir()):
        raise LookupError("no such audio file")
    return p, key[2:]


def list_sources() -> list[dict]:
    out = []
    d = sources_dir()
    if os.path.isdir(d):
        for n in sorted(os.listdir(d), key=str.lower):
            p = os.path.join(d, n)
            if os.path.isfile(p) and os.path.splitext(n)[1].lower() in AUDIO_EXTS:
                out.append({"key": f"a:{n}", "name": n, "kind": "audio file", "size": os.path.getsize(p)})
    for v in library.list_videos():
        i = v.get("info") or {}
        if i.get("has_audio"):
            out.append({"key": f"v:{v['id']}", "name": v["name"], "kind": "video", "duration": i.get("duration")})
    return out


def add_source(filename: str, stream) -> str:
    """Save an uploaded audio file as a source; returns its key."""
    base = os.path.basename(filename or "")
    if os.path.splitext(base)[1].lower() not in AUDIO_EXTS:
        raise ValueError(f"unsupported audio file: {base}")
    os.makedirs(sources_dir(), exist_ok=True)
    from ..fileops import _free
    dst = _free(os.path.join(sources_dir(), _safe(os.path.splitext(base)[0]) + os.path.splitext(base)[1].lower()))
    with open(dst, "wb") as f:
        shutil.copyfileobj(stream, f)
    return "a:" + os.path.basename(dst)


def _version(path: str) -> str:
    st = os.stat(path)
    return f"{st.st_size}:{st.st_mtime_ns}"


def _ckey(key: str) -> str:
    return "snd:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


_preview_locks: dict[str, threading.Lock] = {}
_preview_guard = threading.Lock()


def preview_audio(key: str) -> str:
    """A browser-friendly copy of the source's audio (AAC in .m4a), made once and cached.

    Browsers ask for a media file several times at once, so the copy is built under a
    per-source lock into a private temporary file and only then moved into place: two
    concurrent builds used to interleave into one corrupt file that played only partway."""
    path, _ = source_path(key)
    out = os.path.join(_cache_dir(), f"{_ckey(key)[4:]}-{hashlib.sha1(_version(path).encode()).hexdigest()[:8]}-v2.m4a")
    if os.path.exists(out):
        return out
    with _preview_guard:
        lock = _preview_locks.setdefault(out, threading.Lock())
    with lock:
        if not os.path.exists(out):  # another request may have finished it while we waited
            os.makedirs(_cache_dir(), exist_ok=True)
            tmp = f"{out}.{os.getpid()}-{threading.get_ident()}.part.m4a"
            try:
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-map", "0:a:0", "-ac", "2",
                                "-ar", "48000", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", tmp],
                               check=True, capture_output=True, **_NOWIN)
                os.replace(tmp, out)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
    return out


# ==== analysis ======================================================================================

def _pcm_blocks(stream, nbytes: int):
    """float32 sample blocks from a byte stream. A pipe read can return any byte count (it
    does when decoding is slower than reading, as with audio inside video files), so the odd
    bytes of a sample split across two reads are carried over, never dropped: dropping them
    shifts every later sample and turns the rest of the file into noise."""
    pending = b""
    while True:
        buf = stream.read(nbytes)
        if not buf:
            break
        buf = pending + buf
        usable = len(buf) // 4 * 4
        pending = buf[usable:]
        if usable:
            yield np.nan_to_num(np.frombuffer(buf[:usable], np.float32), nan=0.0, posinf=1.0, neginf=-1.0)


def _features(path: str, job=None) -> dict:
    """Stream the audio as mono float32 at SR; per 10 ms hop: level (dBFS), min, max, onset strength."""
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(SR),
                             "-f", "f32le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **_NOWIN)
    win = np.hanning(WIN).astype(np.float32)
    levels, lo, hi, flux = [], [], [], []
    carry = np.zeros(0, np.float32)
    prev_spec = None
    chunk = SR * 30 // HOP * HOP  # 30 s at a time
    try:
        for block in _pcm_blocks(proc.stdout, chunk * 4):
            x = np.concatenate([carry, block])
            n = (len(x) - (WIN - HOP)) // HOP if len(x) >= WIN else 0
            if n <= 0:
                carry = x
                continue
            hops = x[: n * HOP].reshape(n, HOP)
            levels.append(10 * np.log10(np.mean(hops ** 2, axis=1) + 1e-12))
            lo.append(hops.min(axis=1))
            hi.append(hops.max(axis=1))
            frames = np.lib.stride_tricks.sliding_window_view(x[: (n - 1) * HOP + WIN], WIN)[::HOP][:n] * win
            spec = np.log1p(np.abs(np.fft.rfft(frames, axis=1)) * 10)
            first = spec[:1] if prev_spec is None else prev_spec
            d = np.diff(np.concatenate([first, spec]), axis=0)
            flux.append(np.maximum(d, 0).sum(axis=1))
            prev_spec = spec[-1:]
            carry = x[n * HOP:]
            if job:
                job.check()
                job.update(message=f"analysing {sum(len(a) for a in levels) * HOP / SR:.0f} s")
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if carry.size:  # the last partial hop
        levels.append([10 * np.log10(np.mean(carry ** 2) + 1e-12)])
        lo.append([carry.min()])
        hi.append([carry.max()])
        flux.append([0.0])
    if not levels:
        raise ValueError("no audio could be decoded")
    return {"level": np.concatenate(levels).astype(np.float32), "lo": np.concatenate(lo).astype(np.float32),
            "hi": np.concatenate(hi).astype(np.float32), "flux": np.concatenate(flux).astype(np.float32)}


def analyze(job, key: str) -> dict:
    path, name = source_path(key)
    f = _load_features(key, path)
    if f is None:
        f = _features(path, job)
        buf = io.BytesIO()
        np.savez_compressed(buf, **f)
        cache_put(_ckey(key), "sound-features-v2", _version(path), buf.getvalue())
    return {"key": key, "name": name, "duration": round(len(f["level"]) * HOP / SR, 3)}


def _load_features(key: str, path: str) -> dict | None:
    raw = cache_get(_ckey(key), "sound-features-v2", _version(path))
    if raw is None:
        return None
    with np.load(io.BytesIO(raw)) as z:
        return {k: z[k] for k in z.files}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(x, np.ones(n) / n, mode="same") if len(x) > n else x


def detect(f: dict, below: float = 35.0, min_gap: float = 0.3, min_sound: float = 0.05) -> dict:
    """Gaps, sounds and onsets from cached features.
    below: a gap is quieter than (the file's loud level - below) dB, and never above -20 dBFS.
    min_gap: shorter quiet spans don't split a sound. min_sound: shorter sounds are ignored."""
    lv = _smooth(f["level"].astype(np.float64), 3)
    audible = lv[lv > -100]
    loud = float(np.percentile(audible, 95)) if audible.size else -100.0
    floor = float(np.percentile(audible, 10)) if audible.size else -100.0
    thr = min(loud - below, -20.0)
    quiet = lv < thr
    hop_s = HOP / SR
    gaps = [(a, b) for a, b in _runs(quiet) if (b - a) * hop_s >= min_gap]
    # sounds: everything between the gaps that clears min_sound
    loud_mask = np.ones(len(lv), bool)
    for a, b in gaps:
        loud_mask[a:b] = False
    sounds = []
    for a, b in _runs(loud_mask):
        if (b - a) * hop_s < min_sound:
            continue
        seg = lv[a:b]
        sounds.append({"start": round(a * hop_s, 3), "end": round(b * hop_s, 3),
                       "level": round(float(10 * np.log10(np.mean(10 ** (seg / 10)) + 1e-12)), 1),
                       "peak": round(float(20 * np.log10(max(np.abs(f["lo"][a:b]).max(), np.abs(f["hi"][a:b]).max()) + 1e-9)), 1)})
    # onsets: peaks of onset strength well above its local median, in audible parts
    fl = f["flux"].astype(np.float64)
    med = np.array([np.median(fl[max(0, i - 25):i + 25]) for i in range(0, len(fl))]) if len(fl) < 20000 \
        else _smooth(fl, 51)
    mad = float(np.median(np.abs(fl - np.median(fl)))) + 1e-9
    strong = 0.15 * (float(np.percentile(fl, 99.5)) - float(np.median(fl)))  # ignores the texture of noise beds
    # a hit's peak often lands in the frame just before the level rises, so look one or two hops ahead
    audible_soon = ~quiet | np.concatenate([~quiet[1:], [False]]) | np.concatenate([~quiet[2:], [False, False]])
    cand = (fl > med + max(4 * mad, strong)) & audible_soon
    onsets, last = [], -10
    for i in np.nonzero(cand)[0]:
        lo_i, hi_i = max(0, i - 3), min(len(fl), i + 4)
        if fl[i] == fl[lo_i:hi_i].max() and i - last >= 5:
            onsets.append(round(float(i * hop_s), 3))
            last = i
    return {"threshold": round(thr, 1), "loud": round(loud, 1), "floor": round(floor, 1),
            "gaps": [[round(a * hop_s, 3), round(b * hop_s, 3)] for a, b in gaps],
            "sounds": sounds, "onsets": onsets}


def waveform(f: dict, points: int = 60000) -> dict:
    """Min/max peaks for drawing, pooled down to at most `points` columns."""
    n = len(f["lo"])
    k = max(1, int(np.ceil(n / points)))
    m = n // k * k
    lo = f["lo"][:m].reshape(-1, k).min(axis=1)
    hi = f["hi"][:m].reshape(-1, k).max(axis=1)
    lv = f["level"][:m].reshape(-1, k).max(axis=1)
    return {"step": k * HOP / SR, "lo": np.round(lo, 3).tolist(), "hi": np.round(hi, 3).tolist(),
            "level": np.round(lv, 1).tolist()}


def analysis(key: str, below: float = 35.0, min_gap: float = 0.3) -> dict:
    path, name = source_path(key)
    f = _load_features(key, path)
    if f is None:
        raise LookupError("not analysed yet")
    out = detect(f, below, min_gap)
    out.update(key=key, name=name, duration=round(len(f["level"]) * HOP / SR, 3), wave=waveform(f),
               below=below, min_gap=min_gap)
    return out


def _tight(f: dict, start: float, end: float, below: float, pad: float = 0.02) -> tuple[float, float]:
    """Shrink [start, end] to the audible part, keeping `pad` seconds each side."""
    thr = detect(f, below)["threshold"]
    hop_s = HOP / SR
    a, b = int(start / hop_s), max(int(start / hop_s) + 1, int(np.ceil(end / hop_s)))
    idx = np.nonzero(f["level"][a:b] >= thr)[0]
    if not idx.size:
        return start, end
    return max(start, (a + idx[0]) * hop_s - pad), min(end, (a + idx[-1] + 1) * hop_s + pad)


# ==== saving clips ====================================================================================

def _probe_audio(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=channels,sample_rate:format=duration", "-of", "json", path],
                         capture_output=True, text=True, **_NOWIN).stdout
    d = json.loads(out or "{}")
    s = (d.get("streams") or [{}])[0]
    return {"channels": int(s.get("channels") or 2), "duration": float((d.get("format") or {}).get("duration") or 0)}


def _measure(path: str) -> dict:
    """Level (mean power, dBFS), peak (dBFS) and a 120-point envelope of a saved clip."""
    # every channel at the file's own rate: a mono mix-down or resampling would misreport the peak
    ch = max(1, _probe_audio(path)["channels"])
    proc = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-"], capture_output=True, **_NOWIN)
    x = np.frombuffer(proc.stdout[: len(proc.stdout) // (4 * ch) * 4 * ch], np.float32).reshape(-1, ch)
    if not x.size:
        return {"level": None, "peak": None, "envelope": []}
    x = np.abs(x).max(axis=1) * np.sign(x[:, 0] + 1e-12)  # loudest channel per sample, for peak and envelope
    k = max(1, len(x) // 120)
    env = np.abs(x[: len(x) // k * k]).reshape(-1, k).max(axis=1)[:120]
    return {"level": round(float(10 * np.log10(np.mean(x ** 2) + 1e-12)), 1),
            "peak": round(float(20 * np.log10(np.abs(x).max() + 1e-9)), 1),
            "envelope": np.round(env / (env.max() + 1e-9), 3).tolist()}


def save(key: str, start: float, end: float, name: str, category: str, tags: list[str] | None = None,
         loop: bool = False, trim: bool = True, fade: float = 0.005, normalize: bool = False,
         below: float = 35.0, notes: str = "") -> dict:
    """Cut [start, end] from a source into the sound library as a 48 kHz / 24-bit WAV."""
    path, src_name = source_path(key)
    start, end = float(start), float(end)
    if end - start < 0.02:
        raise ValueError("the selection is too short")
    category = _safe(category or "other")
    if trim:
        f = _load_features(key, path)
        if f is not None:
            start, end = _tight(f, start, end, below)
    dur = end - start
    fade = max(0.0, min(float(fade), dur / 4))
    ch = min(2, _probe_audio(path)["channels"])
    af = []
    if fade:
        af += [f"afade=t=in:st=0:d={fade:.4f}", f"afade=t=out:st={dur - fade:.4f}:d={fade:.4f}"]
    if normalize:  # peak to -1 dBFS
        # astats reports at the info log level; its last "Peak level" is the overall one
        m = subprocess.run(["ffmpeg", "-hide_banner", "-v", "info", "-ss", f"{start:.4f}", "-t", f"{dur:.4f}",
                            "-i", path, "-vn", "-map", "0:a:0", "-af", "astats=metadata=0:reset=0", "-f", "null", "-"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", **_NOWIN).stderr
        pk = re.findall(r"Peak level dB:\s*(-?[\d.]+|-inf)", m)
        if pk and pk[-1] != "-inf":
            af.append(f"volume={-1.0 - float(pk[-1]):.2f}dB")
    folder = os.path.join(sounds_dir(), category)
    os.makedirs(folder, exist_ok=True)
    from ..fileops import _free
    out = _free(os.path.join(folder, _safe(name or os.path.splitext(src_name)[0]) + ".wav"))
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.4f}", "-t", f"{dur:.4f}", "-i", path, "-vn",
           "-map", "0:a:0", "-ac", str(ch), "-ar", str(OUT_RATE)]
    if af:
        cmd += ["-af", ",".join(af)]
    subprocess.run(cmd + ["-c:a", "pcm_s24le", out], check=True, capture_output=True, **_NOWIN)
    return _index(out, name=name or os.path.splitext(os.path.basename(out))[0], category=category,
                  tags=tags or [], loop=loop, source=key, source_name=src_name, start=start, end=end, notes=notes)


def _index(path: str, name: str, category: str, tags: list[str], loop: bool = False, source: str | None = None,
           source_name: str | None = None, start: float | None = None, end: float | None = None,
           notes: str = "") -> dict:
    m = _measure(path)
    a = _probe_audio(path)
    st = os.stat(path)
    with _db() as c:
        cur = c.execute(
            "INSERT OR REPLACE INTO sounds (path, name, category, tags, duration, level, peak, channels, loop, "
            "source, source_name, source_start, source_end, notes, envelope, size, mtime, created) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (os.path.abspath(path), name, category, json.dumps([t for t in tags if t]), a["duration"], m["level"],
             m["peak"], a["channels"], int(bool(loop)), source, source_name,
             None if start is None else round(start, 3), None if end is None else round(end, 3), notes,
             json.dumps(m["envelope"]), st.st_size, st.st_mtime, time.time()))
        sid = cur.lastrowid
    return get(sid)


# ==== the library ======================================================================================

def _row(r) -> dict:
    d = dict(r)
    d["tags"] = json.loads(d["tags"] or "[]")
    d["envelope"] = json.loads(d["envelope"] or "[]")
    d["loop"], d["fav"] = bool(d["loop"]), bool(d["fav"])
    d["file"] = os.path.basename(d["path"])
    return d


def get(sid: int) -> dict:
    with _db() as c:
        r = c.execute("SELECT * FROM sounds WHERE id=?", (int(sid),)).fetchone()
    if not r:
        raise LookupError("no such sound")
    return _row(r)


def list_sounds(category: str | None = None, q: str | None = None, fav: bool = False) -> list[dict]:
    sql, args = "SELECT * FROM sounds WHERE 1=1", []
    if category:
        sql += " AND category=?"
        args.append(category)
    if fav:
        sql += " AND fav=1"
    with _db() as c:
        rows = [_row(r) for r in c.execute(sql + " ORDER BY created DESC", args)]
    rows = [r for r in rows if os.path.exists(r["path"])]
    if q:
        words = q.lower().split()
        rows = [r for r in rows if all(w in " ".join([r["name"], r["category"] or "", " ".join(r["tags"]),
                                                      r["source_name"] or "", r["notes"] or ""]).lower()
                                       for w in words)]
    return rows


def update(sid: int, **fields) -> dict:
    s = get(sid)
    sets, args = [], []
    for k in ("name", "notes"):
        if k in fields and fields[k] is not None:
            sets.append(f"{k}=?")
            args.append(str(fields[k])[:500])
    for k in ("loop", "fav"):
        if k in fields and fields[k] is not None:
            sets.append(f"{k}=?")
            args.append(int(bool(fields[k])))
    if fields.get("tags") is not None:
        sets.append("tags=?")
        args.append(json.dumps([str(t).strip() for t in fields["tags"] if str(t).strip()][:50]))
    if fields.get("category") and _safe(fields["category"]) != s["category"]:
        cat = _safe(fields["category"])  # keep the folder layout in step with the category
        folder = os.path.join(sounds_dir(), cat)
        os.makedirs(folder, exist_ok=True)
        from ..fileops import _free
        dst = _free(os.path.join(folder, os.path.basename(s["path"])))
        shutil.move(s["path"], dst)
        sets += ["category=?", "path=?"]
        args += [cat, os.path.abspath(dst)]
    if sets:
        with _db() as c:
            c.execute(f"UPDATE sounds SET {', '.join(sets)} WHERE id=?", args + [int(sid)])
    return get(sid)


def delete(sid: int) -> None:
    """Move the file to the sound folder's _trash (recoverable by hand) and forget it."""
    s = get(sid)
    trash = os.path.join(sounds_dir(), "_trash")
    os.makedirs(trash, exist_ok=True)
    if os.path.exists(s["path"]):
        from ..fileops import _free
        shutil.move(s["path"], _free(os.path.join(trash, os.path.basename(s["path"]))))
    with _db() as c:
        c.execute("DELETE FROM sounds WHERE id=?", (int(sid),))


def rescan(job=None) -> dict:
    """Pick up audio files added to the sound folder by hand; forget ones that were removed."""
    base = sounds_dir()
    os.makedirs(base, exist_ok=True)
    with _db() as c:
        known = {config.norm(r["path"]): r["id"] for r in c.execute("SELECT id, path FROM sounds")}
    found, added = set(), 0
    for dp, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(("_", "."))]
        for n in files:
            if os.path.splitext(n)[1].lower() not in AUDIO_EXTS:
                continue
            p = os.path.join(dp, n)
            found.add(config.norm(p))
            if config.norm(p) in known:
                continue
            rel = os.path.relpath(dp, base)
            cat = "other" if rel in (".", "") else rel.split(os.sep)[0]
            try:
                _index(p, name=os.path.splitext(n)[0], category=cat, tags=[])
                added += 1
            except Exception:
                continue
            if job:
                job.update(message=f"added {added}")
    gone = [sid for p, sid in known.items() if p not in found]
    with _db() as c:
        c.executemany("DELETE FROM sounds WHERE id=?", [(g,) for g in gone])
    return {"added": added, "removed": len(gone)}
