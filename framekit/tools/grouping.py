"""Group videos by the similarity of their sampled frames.

Similarity between two videos is set-to-set: each sampled frame of A is
matched to its most similar frame of B, those best scores are averaged, and
the result is symmetrized. Two clips that share only some shots still score
as related, which averaging whole videos into one vector would miss.

Groups come from average-linkage hierarchical clustering cut at a distance
threshold, so the number of groups is never guessed. The pairwise distance
matrix is saved with the run, so changing the threshold regroups instantly.

A grouping run lives in work/groupings/<run>/ (dist.npy + run.json). Manual
edits (moving a video, renaming a group) are stored in run.json; re-cutting at
a new threshold replaces them.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import secrets
import shutil
import time

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .. import config, fileops, library, samples

DEFAULT_THRESHOLD = {"visual": 0.20, "clip": 0.10}
RUN_RE = re.compile(r"^[0-9]{8}-[0-9]{6}(-[a-z0-9]+)?$")


def _dir(run_id: str) -> str:
    if not RUN_RE.match(run_id):
        raise ValueError("bad grouping id")
    return os.path.join(config.WORK, "groupings", run_id)


def similarity_matrix(mats: list[np.ndarray]) -> np.ndarray:
    """Symmetric set-to-set similarity from per-video (k_i, d) normalized embeddings."""
    n = len(mats)
    F = np.concatenate(mats).astype(np.float32)
    starts = np.cumsum([0] + [len(m) for m in mats[:-1]])
    A = np.zeros((n, n), np.float32)
    for i, m in enumerate(mats):
        s = m @ F.T                                  # (k_i, total frames)
        best = np.maximum.reduceat(s, starts, axis=1)  # (k_i, n): best match per video
        A[i] = best.mean(axis=0)
    S = (A + A.T) / 2
    np.fill_diagonal(S, 1.0)
    return S


def build(job, under: list[str] | None = None, video_ids: list[str] | None = None,
          backend: str = "visual", threshold: float | None = None) -> dict:
    videos = library.list_videos(under)
    if video_ids:
        wanted = set(video_ids)
        videos = [v for v in videos if v["id"] in wanted]
    if len(videos) < 2:
        raise ValueError("need at least two videos to group")
    got = samples.ensure(videos, job)
    S = got["samples"]
    videos = [v for v in videos if v["id"] in S]
    job.update(message=f"embedding samples ({backend})", progress=0, total=len(videos))
    mats = []
    for i, v in enumerate(videos, 1):
        job.check()
        mats.append(samples.embeddings(v, S[v["id"]], backend))
        job.update(progress=i)
    job.update(message="comparing videos")
    dist = np.clip(1.0 - similarity_matrix(mats), 0, 2).astype(np.float32)
    np.fill_diagonal(dist, 0)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    d = _dir(run_id)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "dist.npy"), dist)
    run = {"id": run_id, "created": time.time(), "backend": backend,
           "videos": [v["id"] for v in videos],
           "errors": {k: e for k, e in got["errors"].items()},
           "threshold": threshold if threshold is not None else DEFAULT_THRESHOLD.get(backend, 0.12)}
    _cut(run, dist)
    _save(run)
    return {"run": run_id, "groups": sum(1 for g in run["groups"] if g["id"] != "singles")}


def _cut(run: dict, dist: np.ndarray) -> None:
    ids = run["videos"]
    n = len(ids)
    labels = (fcluster(linkage(squareform(dist, checks=False), "average"),
                       t=run["threshold"], criterion="distance") if n > 1 else np.array([1]))
    buckets: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        buckets.setdefault(int(lab), []).append(i)
    multi = sorted((b for b in buckets.values() if len(b) > 1), key=len, reverse=True)
    singles = [b[0] for b in buckets.values() if len(b) == 1]
    groups = []
    for k, b in enumerate(multi, 1):
        sub = dist[np.ix_(b, b)]
        order = [b[j] for j in np.argsort(sub.mean(axis=1))]  # medoid first
        groups.append({"id": f"g{k}", "name": f"Group {k}", "members": [ids[j] for j in order],
                       "spread": round(float(sub[np.triu_indices(len(b), 1)].mean()), 4)})
    groups.append({"id": "singles", "name": "No close match", "members": [ids[j] for j in singles]})
    run["groups"] = groups
    run["edited"] = False


def _save(run: dict) -> None:
    with open(os.path.join(_dir(run["id"]), "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f)


def load(run_id: str) -> dict:
    p = os.path.join(_dir(run_id), "run.json")
    if not os.path.exists(p):
        raise LookupError("no such grouping")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_runs() -> list[dict]:
    base = os.path.join(config.WORK, "groupings")
    out = []
    if os.path.isdir(base):
        for rid in os.listdir(base):
            try:
                r = load(rid)
            except (LookupError, ValueError):
                continue
            out.append({"id": r["id"], "created": r["created"], "backend": r["backend"],
                        "videos": len(r["videos"]), "threshold": r["threshold"],
                        "groups": sum(1 for g in r["groups"] if g["id"] != "singles")})
    return sorted(out, key=lambda r: r["created"], reverse=True)


def delete(run_id: str) -> None:
    shutil.rmtree(_dir(run_id), ignore_errors=True)


def recut(run_id: str, threshold: float) -> dict:
    run = load(run_id)
    run["threshold"] = float(threshold)
    _cut(run, np.load(os.path.join(_dir(run_id), "dist.npy")))
    _save(run)
    return run


def distances(run_id: str) -> tuple[list[str], np.ndarray]:
    run = load(run_id)
    return run["videos"], np.load(os.path.join(_dir(run_id), "dist.npy"))


def move(run_id: str, vid: str, to: str | None) -> dict:
    """Move a video to group `to` (None or "new" = a new group of its own)."""
    run = load(run_id)
    if vid not in run["videos"]:
        raise ValueError("video is not in this grouping")
    for g in run["groups"]:
        if vid in g["members"]:
            g["members"].remove(vid)
    if to in (None, "", "new"):
        k = 1 + max([int(g["id"][1:]) for g in run["groups"] if g["id"].startswith("g")] or [0])
        run["groups"].insert(-1, {"id": f"g{k}", "name": f"Group {k}", "members": [vid]})
    else:
        target = next((g for g in run["groups"] if g["id"] == to), None)
        if not target:
            raise ValueError("no such group")
        target["members"].append(vid)
    run["groups"] = [g for g in run["groups"] if g["members"] or g["id"] == "singles"]
    run["edited"] = True
    _save(run)
    return run


def rename(run_id: str, gid: str, name: str) -> dict:
    run = load(run_id)
    name = name.strip()[:80]
    if not name:
        raise ValueError("empty name")
    for g in run["groups"]:
        if g["id"] == gid:
            g["name"] = name
    run["edited"] = True
    _save(run)
    return run


def export(run_id: str, fmt: str = "json") -> tuple[str, str]:
    """(content, mimetype) listing every video with its group."""
    run = load(run_id)
    rows = []
    for g in run["groups"]:
        for vid in g["members"]:
            v = library.get(vid) or {"path": vid, "name": vid}
            rows.append({"group": g["name"], "group_id": g["id"], "video_id": vid,
                         "name": v["name"], "path": v["path"]})
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["group"])
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue(), "text/csv"
    return json.dumps({"grouping": run_id, "backend": run["backend"], "threshold": run["threshold"],
                       "groups": [{"name": g["name"], "videos": [r for r in rows if r["group_id"] == g["id"]]}
                                  for g in run["groups"]]}, indent=2), "application/json"


def _safe(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .") or "group"


def organize(job, run_id: str, mode: str = "copy", include_singles: bool = False) -> dict:
    """Materialize the grouping as folders.

    copy: work/exports/groups-<run>/<group>/<file>. Originals untouched.
    move: <file's own root>/_groups/<run>/<group>/<file>, logged as an
          undoable batch (same undo as the quarantine).
    """
    if mode not in ("copy", "move"):
        raise ValueError("mode must be copy or move")
    run = load(run_id)
    groups = [g for g in run["groups"] if include_singles or g["id"] != "singles"]
    total = sum(len(g["members"]) for g in groups)
    job.update(message=f"{mode} into folders", progress=0, total=total)
    batch = fileops.new_batch() if mode == "move" else None
    dest_base = os.path.join(config.EXPORTS, f"groups-{run_id}")
    done, failed, remap = 0, {}, {}
    for g in groups:
        for vid in g["members"]:
            job.check()
            v = library.get(vid)
            if not v or not v["present"] or not os.path.exists(v["path"]):
                failed[vid] = "file missing"
                continue
            try:
                if mode == "copy":
                    dst = fileops._free(os.path.join(dest_base, _safe(g["name"]), v["name"]))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(v["path"], dst)
                else:
                    dst = os.path.join(v["root"], "_groups", run_id, _safe(g["name"]), v["name"])
                    dst = fileops.move_logged(v["path"], dst, batch, "organize", f"group {g['name']}")
                    library.mark_missing([vid])
                    remap[vid] = library.add_path(dst, root=v["root"])["id"]
                done += 1
            except OSError as e:
                failed[vid] = str(e)
            job.update(progress=done + len(failed))
    if remap:  # moved files have new paths, hence new ids
        run["videos"] = [remap.get(x, x) for x in run["videos"]]
        for g in run["groups"]:
            g["members"] = [remap.get(x, x) for x in g["members"]]
        _save(run)
    return {"mode": mode, "done": done, "failed": failed, "batch": batch,
            "folder": dest_base if mode == "copy" else None}
