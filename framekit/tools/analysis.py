"""Frame analysis within one extracted frame set (a frames run).

analyze() computes, once per run, an embedding and perceptual hash per frame
(from the thumbnails) and stores them in <run>/analysis/. Everything after
that is cheap and re-runs instantly when a threshold changes:

  shots     cuts where adjacent frames differ sharply (meaningful for dense
            runs: all, nth, fps)
  dupes     frames within `dup_threshold` Hamming bits of an earlier kept
            frame; excluded from exports by default
  clusters  average-linkage clustering of frame embeddings, cut at a
            distance threshold, numbered by first appearance
  order     chronological | cluster (grouped, then by time) | chain (walk
            to the nearest unvisited frame each step) | custom (drag order)

Exports: ordered zip, one folder per cluster zip, contact sheet, and an
ordered video render (no audio: reordered frames no longer match it).
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .. import config, features, media
from . import frames as frames_tool

MAX_FRAMES = 6000
DEFAULTS = {"visual": {"shot_threshold": 0.35, "cluster_threshold": 0.25, "dup_threshold": 6},
            "clip": {"shot_threshold": 0.15, "cluster_threshold": 0.15, "dup_threshold": 6}}
ORDER_MODES = ("chronological", "cluster", "chain", "custom")


def _adir(vid: str, run: str) -> str:
    return os.path.join(frames_tool.run_dir(vid, run), "analysis")


def _img_path(vid: str, run: str, m: dict, fr: dict) -> str:
    d = frames_tool.run_dir(vid, run)
    t = os.path.join(d, "thumbs", f"t{fr['n']:06d}.jpg")
    return t if os.path.exists(t) else os.path.join(d, fr["file"])


def analyze(job, vid: str, run: str, backend: str = "visual", **params) -> dict:
    m = frames_tool.get_run(vid, run)
    n = len(m["frames"])
    if n > MAX_FRAMES:
        raise ValueError(f"{n} frames is too many to analyze (max {MAX_FRAMES}); "
                         "extract with scene, fps or nth mode instead")
    job.update(message="reading frames", progress=0, total=n)
    embs, hashes = [], []
    batch = []
    for i, fr in enumerate(m["frames"], 1):
        job.check()
        with Image.open(_img_path(vid, run, m, fr)) as im:
            img = im.convert("RGB")
        hashes.append(features.phash_image(img)[0])
        batch.append(img)
        if len(batch) == 64 or i == n:
            embs.append(features.embed(batch, backend))
            batch = []
        job.update(progress=i)
    d = _adir(vid, run)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "emb.npy"), np.concatenate(embs).astype(np.float32))
    np.save(os.path.join(d, "hash.npy"), np.array(hashes, dtype=np.uint64))
    state = {"backend": backend, "params": {**DEFAULTS.get(backend, DEFAULTS["visual"]),
                                            **{k: v for k, v in params.items() if v is not None}},
             "order_mode": "chronological", "custom": None, "manual_excluded": [], "manual_included": []}
    _compute(vid, run, state)
    return {"frames": n, "shots": len(state["shots"]), "clusters": state["n_clusters"],
            "dupes": sum(1 for x in state["dup_of"] if x is not None)}


def _load_arrays(vid: str, run: str):
    d = _adir(vid, run)
    return np.load(os.path.join(d, "emb.npy")), np.load(os.path.join(d, "hash.npy"))


def _compute(vid: str, run: str, state: dict) -> dict:
    E, H = _load_arrays(vid, run)
    p = state["params"]
    n = len(E)
    # shots
    adj = 1.0 - np.einsum("ij,ij->i", E[1:], E[:-1]) if n > 1 else np.zeros(0)
    cuts = [0] + [i + 1 for i, dd in enumerate(adj) if dd > p["shot_threshold"]] + [n]
    shots = [[cuts[i], cuts[i + 1] - 1] for i in range(len(cuts) - 1)]
    # duplicate frames: compare each frame against the kept ones
    dup_of: list[int | None] = []
    kept: list[int] = []
    for i in range(n):
        if kept:
            dist = features.hamming(H[i], H[kept])
            j = int(np.argmin(dist))
            if dist[j] <= p["dup_threshold"]:
                dup_of.append(kept[j])
                continue
        kept.append(i)
        dup_of.append(None)
    # clusters (over kept frames; duplicates inherit their original's cluster)
    labels = np.zeros(n, dtype=int)
    if len(kept) > 1:
        lab = fcluster(linkage(pdist(E[kept], "cosine"), "average"),
                       t=p["cluster_threshold"], criterion="distance")
        for k, i in enumerate(kept):
            labels[i] = lab[k]
    elif kept:
        labels[kept[0]] = 1
    for i in range(n):
        if dup_of[i] is not None:
            labels[i] = labels[dup_of[i]]
    renum, clusters = {}, []
    for i in range(n):
        renum.setdefault(int(labels[i]), len(renum))
        clusters.append(renum[int(labels[i])])
    state.update(shots=shots, dup_of=dup_of, clusters=clusters, n_clusters=len(renum))
    _save(vid, run, state)
    return state


def _save(vid: str, run: str, state: dict) -> None:
    with open(os.path.join(_adir(vid, run), "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)


def load(vid: str, run: str) -> dict | None:
    p = os.path.join(_adir(vid, run), "state.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        state = json.load(f)
    state["order"] = ordered(vid, run, state, include_excluded=True)
    state["excluded"] = sorted(excluded(state))
    return state


def set_params(vid: str, run: str, **params) -> dict:
    state = load(vid, run)
    if not state:
        raise LookupError("run not analyzed yet")
    for k in ("shot_threshold", "cluster_threshold", "dup_threshold"):
        if params.get(k) is not None:
            state["params"][k] = float(params[k]) if k != "dup_threshold" else int(params[k])
    _compute(vid, run, state)
    return load(vid, run)


def excluded(state: dict) -> set[int]:
    auto = {i for i, d in enumerate(state["dup_of"]) if d is not None}
    return (auto | set(state.get("manual_excluded", []))) - set(state.get("manual_included", []))


def set_excluded(vid: str, run: str, indices: list[int], exclude: bool) -> dict:
    state = load(vid, run)
    me, mi = set(state.get("manual_excluded", [])), set(state.get("manual_included", []))
    for i in indices:
        if exclude:
            me.add(i)
            mi.discard(i)
        else:
            mi.add(i)
            me.discard(i)
    state["manual_excluded"], state["manual_included"] = sorted(me), sorted(mi)
    _save(vid, run, state)
    return load(vid, run)


def _chain(E: np.ndarray, idx: list[int]) -> list[int]:
    if not idx:
        return []
    sub = E[idx]
    sim = sub @ sub.T
    visited = np.zeros(len(idx), bool)
    cur, out = 0, [0]
    visited[0] = True
    for _ in range(len(idx) - 1):
        s = np.where(visited, -np.inf, sim[cur])
        cur = int(np.argmax(s))
        visited[cur] = True
        out.append(cur)
    return [idx[k] for k in out]


def ordered(vid: str, run: str, state: dict, include_excluded: bool = False) -> list[int]:
    n = len(state["clusters"])
    ex = set() if include_excluded else excluded(state)
    mode = state.get("order_mode", "chronological")
    base = [i for i in range(n)]
    if mode == "cluster":
        base.sort(key=lambda i: (state["clusters"][i], i))
    elif mode == "chain":
        E, _ = _load_arrays(vid, run)
        keep = [i for i in base if i not in excluded(state)]
        base = _chain(E, keep) + [i for i in base if i in excluded(state)]
    elif mode == "custom" and state.get("custom"):
        seen = [i for i in state["custom"] if 0 <= i < n]
        base = seen + [i for i in base if i not in set(seen)]
    return [i for i in base if i not in ex]


def set_order(vid: str, run: str, mode: str, custom: list[int] | None = None) -> dict:
    if mode not in ORDER_MODES:
        raise ValueError(f"order must be one of {ORDER_MODES}")
    state = load(vid, run)
    state["order_mode"] = mode
    if mode == "custom":
        if custom is not None:
            state["custom"] = [int(i) for i in custom]
        elif not state.get("custom"):
            state["custom"] = state["order"]
    _save(vid, run, state)
    return load(vid, run)


# --- exports ----------------------------------------------------------------------

def _stem(vid: str, run: str) -> str:
    m = frames_tool.get_run(vid, run)
    return f"{os.path.splitext(m['name'])[0]}-{run}"


def export_zip(vid: str, run: str, layout: str = "ordered") -> str:
    """ordered: 0001_f000123.jpg...; clusters: cluster_01/f000123.jpg..."""
    m = frames_tool.get_run(vid, run)
    state = load(vid, run)
    d = frames_tool.run_dir(vid, run)
    order = ordered(vid, run, state)
    os.makedirs(config.EXPORTS, exist_ok=True)
    out = os.path.join(config.EXPORTS, f"{_stem(vid, run)}-{layout}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for pos, i in enumerate(order, 1):
            fr = m["frames"][i]
            if layout == "clusters":
                name = f"cluster_{state['clusters'][i] + 1:02d}/{fr['file']}"
            else:
                name = f"{pos:05d}_{fr['file']}"
            z.write(os.path.join(d, fr["file"]), name)
        z.writestr("order.json", json.dumps([{"position": p, "frame": m["frames"][i]["n"],
                                               "time": m["frames"][i]["t"],
                                               "cluster": state["clusters"][i] + 1}
                                              for p, i in enumerate(order, 1)], indent=1))
    return out


def contact_sheet(vid: str, run: str, cols: int = 8, cell: int = 240, max_frames: int = 800) -> str:
    m = frames_tool.get_run(vid, run)
    state = load(vid, run)
    order = ordered(vid, run, state)[:max_frames]
    if not order:
        raise ValueError("no frames to lay out")
    with Image.open(_img_path(vid, run, m, m["frames"][order[0]])) as im:
        ch = round(cell * im.height / im.width)
    rows = (len(order) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (ch + 18)), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(order):
        fr = m["frames"][i]
        x, y = (k % cols) * cell, (k // cols) * (ch + 18)
        with Image.open(_img_path(vid, run, m, fr)) as im:
            sheet.paste(im.convert("RGB").resize((cell, ch)), (x, y))
        t = f"  {fr['t']:.2f}s" if fr.get("t") is not None else ""
        draw.text((x + 4, y + ch + 3), f"#{fr['n']}  c{state['clusters'][i] + 1}{t}", fill=(200, 204, 210))
    os.makedirs(config.EXPORTS, exist_ok=True)
    out = os.path.join(config.EXPORTS, f"{_stem(vid, run)}-sheet.jpg")
    sheet.save(out, quality=88)
    return out


def render(job, vid: str, run: str, fps: float = 24.0) -> str:
    """Encode the ordered, non-excluded frames as an H.264 video (no audio)."""
    m = frames_tool.get_run(vid, run)
    state = load(vid, run)
    order = ordered(vid, run, state)
    if not order:
        raise ValueError("no frames to render")
    d = frames_tool.run_dir(vid, run)
    seq = os.path.join(_adir(vid, run), "render-seq")
    shutil.rmtree(seq, ignore_errors=True)
    os.makedirs(seq)
    ext = m["fmt"]
    try:
        job.update(message="linking frames", progress=0, total=len(order))
        for k, i in enumerate(order, 1):
            src = os.path.join(d, m["frames"][i]["file"])
            dst = os.path.join(seq, f"{k:06d}.{ext}")
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        os.makedirs(config.EXPORTS, exist_ok=True)
        out = os.path.join(config.EXPORTS, f"{_stem(vid, run)}-ordered.mp4")
        job.update(message="encoding", progress=0, total=int(len(order) / fps) or 1)
        media.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
                   "-i", os.path.join(seq, f"%06d.{ext}"),
                   "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                  job=job, duration=len(order) / fps)
        return out
    finally:
        shutil.rmtree(seq, ignore_errors=True)
