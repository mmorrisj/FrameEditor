"""Per-video fingerprints, computed once and cached until the file changes.

- samples: N frames at evenly spaced positions (5%..95%, skipping intros and
  outros), saved as 320px JPEGs with their perceptual hashes. Cheap: one
  accurate seek per sample. Used by duplicate detection and grouping, and as
  the preview strip in the UI.
- sequence: perceptual hashes of the whole video at 1 fps. Costs a full
  decode; used only by the deep duplicate check (trims and overlaps).
- embeddings: feature vectors of the samples for a given backend.
"""
from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from . import features, library, media
from .db import cache_get, cache_put

N_SAMPLES = 12
THUMB_W = 320


def positions(duration: float, n: int = N_SAMPLES) -> list[float]:
    if duration <= 0:
        return [0.0]
    return [round(float(p) * duration, 3) for p in np.linspace(0.05, 0.95, n)]


def sample_path(vid: str, i: int) -> str:
    return library.work_dir(vid, "samples", f"s{i:02d}.jpg")


def get(v: dict, n: int = N_SAMPLES) -> dict:
    """{times, hashes (uint64 array), valid (bool array), paths} for a library video."""
    key = library.cache_key(v, n, "s1")
    raw = cache_get(v["id"], "samples", key)
    if raw:
        d = json.loads(raw)
        paths = [sample_path(v["id"], i) for i in range(len(d["times"]))]
        if all(os.path.exists(p) for p in paths):
            return {"times": d["times"], "hashes": np.array([int(h) for h in d["hashes"]], dtype=np.uint64),
                    "valid": np.array(d["valid"], dtype=bool), "paths": paths}
    times = positions(v["info"]["duration"], n)
    os.makedirs(library.work_dir(v["id"], "samples"), exist_ok=True)
    hashes, valid, paths = [], [], []
    for i, t in enumerate(times):
        img = media.grab_frame(v["path"], t, THUMB_W)
        p = sample_path(v["id"], i)
        img.save(p, quality=85)
        h, ok = features.phash_image(img)
        hashes.append(h)
        valid.append(ok)
        paths.append(p)
    cache_put(v["id"], "samples", key, json.dumps(
        {"times": times, "hashes": [str(h) for h in hashes], "valid": valid}).encode())
    return {"times": times, "hashes": np.array(hashes, dtype=np.uint64),
            "valid": np.array(valid, dtype=bool), "paths": paths}


def ensure(videos: list[dict], job=None, workers: int = 4) -> dict[str, dict]:
    """Samples for many videos in parallel; failures are reported, not fatal."""
    out, errors = {}, {}
    if job:
        job.update(message="sampling frames", progress=0, total=len(videos))

    def one(v):
        try:
            return v["id"], get(v), None
        except Exception as e:
            return v["id"], None, str(e)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (vid, s, err) in enumerate(ex.map(one, videos), 1):
            if s is not None:
                out[vid] = s
            else:
                errors[vid] = err
            if job:
                job.check()
                job.update(progress=i)
    return {"samples": out, "errors": errors}


def sequence(v: dict, job=None, fps: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(hashes uint64 (n,), valid bool (n,)) at `fps` over the whole video."""
    key = library.cache_key(v, fps, "q1")
    raw = cache_get(v["id"], "seq", key)
    if raw:
        z = np.load(io.BytesIO(raw))
        return z["h"], z["v"]
    g = media.gray_sequence(v["path"], fps=fps, size=32, job=job)
    h = features.phash_gray(g) if len(g) else np.zeros(0, np.uint64)
    ok = features.detail_mask(g) if len(g) else np.zeros(0, bool)
    buf = io.BytesIO()
    np.savez(buf, h=h, v=ok)
    cache_put(v["id"], "seq", key, buf.getvalue())
    return h, ok


def embeddings(v: dict, s: dict, backend: str) -> np.ndarray:
    """(n_samples, d) embedding matrix of a video's samples."""
    key = library.cache_key(v, len(s["paths"]), backend, "e1")
    kind = f"emb:{backend}"
    raw = cache_get(v["id"], kind, key)
    if raw:
        return np.load(io.BytesIO(raw))
    imgs = []
    for p in s["paths"]:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    e = features.embed(imgs, backend).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, e)
    cache_put(v["id"], kind, key, buf.getvalue())
    return e
