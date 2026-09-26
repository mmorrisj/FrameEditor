"""Unique frames: a drop folder that thins image sets down to distinct views.

Drop images into the inbox folder (work/frames-inbox by default, or the
"inbox" setting in framekit.json). Each subfolder is its own set, and loose
images directly in the inbox form one more set. Sets never compare against
each other.

Within a set, images are ranked best first (sharpest by default, so a
motion-blurred frame never beats a crisp one of the same view). Walking that
ranking, an image is kept unless it is within `threshold` embedding distance,
or `bits` perceptual-hash bits, of an image already kept. Removed images move
to the quarantine as an undoable batch. Undoing a batch pins those images, so
the watcher never removes them again.

The watcher polls the inbox and processes a set once its contents have
stopped changing for a few seconds. It never trusts file timestamps, because
Windows Explorer stamps a copied file with the source's modified time.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time

import numpy as np
from PIL import Image, ImageOps

from .. import config, features, fileops, jobs
from ..db import cache_get, cache_put

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PREFER = {"sharpest": "Sharpest image", "largest": "Largest image", "name": "First by file name"}
DEFAULT_THRESHOLD = {"visual": 0.10, "clip": 0.015}
DEFAULTS = {"backend": "visual", "threshold": None, "bits": 4, "prefer": "sharpest", "watch": True}
SECTION = "unique"


def _dir(*parts: str) -> str:
    return os.path.join(config.WORK, "unique", *parts)


def inbox() -> str:
    p = config.settings(SECTION).get("inbox") or os.environ.get("FRAMEKIT_FRAME_INBOX") \
        or os.path.join(config.WORK, "frames-inbox")
    os.makedirs(p, exist_ok=True)
    return os.path.abspath(p)


# --- settings -----------------------------------------------------------------------

def threshold_for(backend: str) -> float:
    """The saved cutoff for a backend (each backend has its own scale), or its default."""
    saved = config.settings(SECTION).get("thresholds") or {}
    return float(saved.get(backend, DEFAULT_THRESHOLD.get(backend, 0.1)))


def get_settings() -> dict:
    saved = config.settings(SECTION)
    s = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS and k != "threshold"}}
    if s["backend"] not in features.backends():
        s["backend"] = "visual"
    s["threshold"] = threshold_for(s["backend"])
    s["thresholds"] = {b: threshold_for(b) for b in DEFAULT_THRESHOLD}
    return s


def save_settings(**kw) -> dict:
    s = {}
    if kw.get("backend") is not None:
        if not features.backends().get(kw["backend"]):
            raise ValueError(f"backend {kw['backend']!r} is not available")
        s["backend"] = kw["backend"]
    if kw.get("threshold") is not None:
        t = float(kw["threshold"])
        if not 0 <= t <= 1:
            raise ValueError("threshold must be between 0 and 1")
        backend = s.get("backend") or get_settings()["backend"]
        s["thresholds"] = {**(config.settings(SECTION).get("thresholds") or {}), backend: t}
    if kw.get("bits") is not None:
        b = int(kw["bits"])
        if not 0 <= b <= 32:
            raise ValueError("bits must be between 0 and 32")
        s["bits"] = b
    if kw.get("prefer") is not None:
        if kw["prefer"] not in PREFER:
            raise ValueError("unknown preference")
        s["prefer"] = kw["prefer"]
    if kw.get("watch") is not None:
        s["watch"] = bool(kw["watch"])
    config.save_settings(SECTION, s)
    return get_settings()


# --- sets --------------------------------------------------------------------------------

def list_images(folder: str) -> list[str]:
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [os.path.join(folder, n) for n in names
            if not n.startswith(".") and os.path.splitext(n)[1].lower() in IMAGE_EXTS
            and os.path.isfile(os.path.join(folder, n))]


def signature(folder: str) -> str:
    h = hashlib.sha1()
    for p in list_images(folder):
        try:
            st = os.stat(p)
        except OSError:
            continue
        h.update(f"{os.path.basename(p)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def set_path(name: str) -> str:
    """Folder of a set: "" is the inbox itself, anything else one subfolder."""
    if name in ("", None):
        return inbox()
    if name != os.path.basename(name) or name.startswith((".", "_")) or name in ("..",):
        raise ValueError("bad set name")
    p = os.path.join(inbox(), name)
    if not os.path.isdir(p):
        raise LookupError(f"no set named {name!r}")
    return p


def list_sets() -> list[dict]:
    root = inbox()
    out = [{"name": "", "label": "Loose images", "path": root, "count": len(list_images(root))}]
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n)
        if os.path.isdir(p) and not n.startswith((".", "_")):
            out.append({"name": n, "label": n, "path": p, "count": len(list_images(p))})
    return out


# --- pins (images restored by undo are always kept) ----------------------------------------

def _pins() -> set[str]:
    p = _dir("pinned.json")
    if not os.path.exists(p):
        return set()
    with open(p, encoding="utf-8") as f:
        return set(json.load(f))


def _add_pins(paths: list[str]) -> None:
    pins = _pins() | {config.norm(p) for p in paths}
    os.makedirs(_dir(), exist_ok=True)
    with open(_dir("pinned.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(pins), f)


# --- features ----------------------------------------------------------------------------------

def _sharpness(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def _features(paths: list[str], backend: str, job=None) -> tuple[dict, dict]:
    """{path: (emb, hash, sharpness, pixels)} plus {path: error} for unreadable files."""
    out, errors, todo = {}, {}, []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError as e:
            errors[p] = str(e)
            continue
        key, cid = f"{st.st_size}:{st.st_mtime_ns}", "img:" + hashlib.sha1(config.norm(p).encode()).hexdigest()[:16]
        raw = cache_get(cid, f"uniq:{backend}", key)
        if raw:
            z = np.load(io.BytesIO(raw))
            out[p] = (z["e"], np.uint64(z["h"]), float(z["s"]), int(z["px"]))
        else:
            todo.append((p, cid, key))
    if job:
        job.update(message=f"reading {len(todo)} new images", progress=0, total=len(todo))
    for i in range(0, len(todo), 32):
        if job:
            job.check()
        batch, imgs, meta = todo[i:i + 32], [], []
        for p, cid, key in batch:
            try:
                with Image.open(p) as im:
                    px = im.width * im.height
                    im.draft("RGB", (512, 512))  # JPEG: decode at reduced size, much faster
                    small = ImageOps.exif_transpose(im).convert("RGB")
                small.thumbnail((512, 512))
            except Exception as e:  # locked mid-copy, truncated or not an image
                errors[p] = str(e)
                continue
            imgs.append(small)
            meta.append((p, cid, key, px, small))
        if not imgs:
            continue
        embs = features.embed(imgs, backend)
        for (p, cid, key, px, small), e in zip(meta, embs):
            h = features.phash_image(small)[0]
            s = _sharpness(np.asarray(small.convert("L")))
            buf = io.BytesIO()
            np.savez(buf, e=e.astype(np.float32), h=np.uint64(h), s=s, px=px)
            cache_put(cid, f"uniq:{backend}", key, buf.getvalue())
            out[p] = (e.astype(np.float32), np.uint64(h), s, px)
        if job:
            job.update(progress=min(i + 32, len(todo)))
    return out, errors


# --- selection -----------------------------------------------------------------------------------

def plan(job, folder: str, backend: str | None = None, threshold: float | None = None,
         bits: int | None = None, prefer: str | None = None) -> dict:
    """Decide what to keep; moves nothing."""
    s = get_settings()
    backend = backend or s["backend"]
    threshold = threshold_for(backend) if threshold is None else float(threshold)
    bits = s["bits"] if bits is None else int(bits)
    prefer = prefer or s["prefer"]
    paths = list_images(folder)
    F, errors = _features(paths, backend, job)
    usable = [p for p in paths if p in F]
    pins = _pins()
    pinned = [p for p in usable if config.norm(p) in pins]
    if prefer == "sharpest":
        rank = sorted(usable, key=lambda p: (-F[p][2], p))
    elif prefer == "largest":
        rank = sorted(usable, key=lambda p: (-F[p][3], -F[p][2], p))
    else:
        rank = sorted(usable)
    # pinned images (restored by an undo) are always kept, and they are left out of
    # the comparison so they can't push out images that were kept before
    pinset = set(pinned)
    rank = [p for p in rank if p not in pinset]

    job and job.update(message="comparing", progress=0, total=len(rank))
    kept: list[str] = []
    KE = np.zeros((len(rank), F[rank[0]][0].shape[0] if rank else 1), np.float32)
    KH = np.zeros(len(rank), np.uint64)
    removed = []
    for i, p in enumerate(rank):
        e, h, _, _ = F[p]
        if kept:
            k = len(kept)
            d = 1.0 - KE[:k] @ e
            hb = features.hamming(KH[:k], h)
            j = int(np.argmin(d))
            jb = int(np.argmin(hb))
            if d[j] <= threshold or hb[jb] <= bits:
                twin = j if d[j] <= threshold else jb
                removed.append({"path": p, "kept": kept[twin], "distance": round(float(d[twin]), 4),
                                "bits": int(hb[twin])})
                continue
        KE[len(kept)], KH[len(kept)] = e, h
        kept.append(p)
        if job and i % 50 == 0:
            job.update(progress=i)
    return {"folder": folder, "total": len(paths), "kept": kept + pinned, "removed": removed,
            "skipped": errors, "settings": {"backend": backend, "threshold": threshold,
                                            "bits": bits, "prefer": prefer}}


def apply(p: dict) -> dict:
    """Quarantine a plan's removed images as one undoable batch."""
    if not p["removed"]:
        return {"batch": None, "moved": 0, "failed": {}}
    batch = fileops.new_batch()
    base = os.path.join(config.QUARANTINE, batch, "frames")
    root = inbox()
    moved, failed = [], {}
    for r in p["removed"]:
        src = r["path"]
        rel = os.path.relpath(src, root) if config.inside(src, root) else os.path.join(
            os.path.basename(os.path.dirname(src)), os.path.basename(src))
        try:
            dst = fileops.move_logged(src, os.path.join(base, rel), batch, "unique",
                                      f"near duplicate of {r['kept']}")
            moved.append({**r, "dst": dst})
        except OSError as e:  # e.g. open in an image viewer on Windows
            failed[src] = str(e)
    os.makedirs(_dir("runs"), exist_ok=True)
    with open(_dir("runs", f"{batch}.json"), "w", encoding="utf-8") as f:
        json.dump({"batch": batch, "created": time.time(), "folder": p["folder"], "total": p["total"],
                   "kept": len(p["kept"]), "pairs": moved, "settings": p["settings"]}, f)
    return {"batch": batch, "moved": len(moved), "failed": failed}


def process(job, folder: str, dry_run: bool = False, **kw) -> dict:
    p = plan(job, folder, **kw)
    res = {"folder": folder, "total": p["total"], "kept": len(p["kept"]), "removed": len(p["removed"]),
           "skipped": p["skipped"], "batch": None}
    if dry_run:
        res["pairs"] = p["removed"][:1000]
        return res
    res.update(apply(p))
    return res


def runs(limit: int = 30) -> list[dict]:
    d = _dir("runs")
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d), reverse=True)[:limit]:
        with open(os.path.join(d, name), encoding="utf-8") as f:
            r = json.load(f)
        state = {m["dst"]: m for m in fileops.batch_files(r["batch"])}
        for pr in r["pairs"]:
            m = state.get(pr["dst"], {})
            pr["state"] = "restored" if m.get("restored") else "deleted" if m.get("purged") else "held"
        out.append(r)
    return out


def undo(batch: str) -> dict:
    """Restore a batch and pin its images so they are never removed again."""
    files = fileops.batch_files(batch)
    if not files or any(f["label"] != "unique" for f in files):
        raise LookupError("not a unique-frames batch")
    res = fileops.undo(batch)
    _add_pins([f["src"] for f in fileops.batch_files(batch) if f["restored"]])
    return res


def thumb(path: str) -> str:
    """Cached 240px JPEG of an image inside the inbox or the quarantine."""
    if not (config.inside(path, inbox()) or config.inside(path, config.QUARANTINE)):
        raise ValueError("path outside the inbox and quarantine")
    st = os.stat(path)
    key = hashlib.sha1(f"{config.norm(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()
    out = _dir("thumbs", f"{key}.jpg")
    if not os.path.exists(out):
        os.makedirs(_dir("thumbs"), exist_ok=True)
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((240, 240))
            im.save(out, quality=80)
    return out


# --- watcher ------------------------------------------------------------------------------------------

class Watcher:
    """Processes a set once its listing is unchanged across two polls."""

    def __init__(self, interval: float = 3.0):
        self.interval = interval
        self.seen: dict[str, str] = {}   # folder -> signature at last poll
        self.done: dict[str, str] = {}   # folder -> signature after last processing
        self.last = None                 # last result, for the UI
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> "Watcher":
        if not self._thread:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="unique-watcher")
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.poll()
            except Exception as e:  # keep watching whatever happens
                self.last = {"error": str(e), "at": time.time()}

    def poll(self, run=None) -> list[str]:
        """Check every set; submit jobs (or call `run`) for settled, changed ones."""
        if not get_settings()["watch"] and run is None:
            return []
        started = []
        for st in list_sets():
            path = st["path"]
            if not st["count"]:
                continue
            sig = signature(path)
            settled = self.seen.get(path) == sig
            self.seen[path] = sig
            if not settled or self.done.get(path) == sig:
                continue
            if jobs.active_for(f"unique:{path}"):
                continue
            label = f"Unique frames: {st['label']}"
            self.done[path] = sig  # don't resubmit while it runs; the job records the result
            if run:
                run(self._job, path, label)
            else:
                jobs.submit("unique", label, self._job, path, ref=f"unique:{path}")
            started.append(path)
        return started

    def _job(self, job, path: str) -> dict:
        res = process(job, path)
        self.done[path] = signature(path)
        self.seen[path] = self.done[path]
        self.last = {**{k: v for k, v in res.items() if k != "skipped"},
                     "skipped": len(res["skipped"]), "at": time.time()}
        return res


_watcher: Watcher | None = None


def watcher() -> Watcher:
    global _watcher
    if _watcher is None:
        _watcher = Watcher().start()
    return _watcher
