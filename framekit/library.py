"""The video library: every video file under the configured roots, indexed in
SQLite with its probe results.

Videos are identified by a short hash of their absolute path, so an id is
stable across rescans and a file restored from quarantine gets its old id (and
cached features) back. Probe results are reused while size and mtime match.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import config, media
from .db import db


def video_id(path: str) -> str:
    return hashlib.sha1(config.norm(path).encode("utf-8")).hexdigest()[:12]


def _row(r) -> dict:
    d = dict(r)
    d["info"] = json.loads(d["info"]) if d.get("info") else None
    d["name"] = os.path.basename(d["path"])
    return d


def _walk(root: str):
    skip = {config.norm(config.WORK)} - {config.norm(config.UPLOADS)}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")
                       and config.norm(os.path.join(dirpath, d)) not in skip]
        for f in filenames:
            if os.path.splitext(f)[1].lower() in config.VIDEO_EXTS and not f.startswith("."):
                yield os.path.join(dirpath, f)


def scan(job=None, roots: list[dict] | None = None) -> dict:
    """Index every video under `roots` (default: configured roots).

    New or changed files are probed in parallel; files that vanished from a
    scanned root are marked not present (kept, so their cache survives).
    """
    config.ensure_dirs()
    roots = roots if roots is not None else config.roots()
    found: dict[str, tuple[dict, str, os.stat_result]] = {}
    for r in roots:
        if not os.path.isdir(r["path"]):
            continue
        for p in _walk(r["path"]):
            try:
                found[video_id(p)] = (r, p, os.stat(p))
            except OSError:
                continue
        if job:
            job.check()
            job.update(message=f"found {len(found)} files")

    with db() as c:
        known = {row["id"]: row for row in c.execute("SELECT id, size, mtime, info FROM videos")}
    todo = [vid for vid, (_, _, st) in found.items()
            if vid not in known or known[vid]["size"] != st.st_size
            or known[vid]["mtime"] != st.st_mtime or not known[vid]["info"]]

    def probe_one(vid):
        _, p, _ = found[vid]
        try:
            return vid, media.probe(p), None
        except Exception as e:
            return vid, None, str(e)

    now = time.time()
    results = []
    if job:
        job.update(message="probing", progress=0, total=len(todo))
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, res in enumerate(ex.map(probe_one, todo), 1):
            results.append(res)
            if job:
                job.check()
                job.update(progress=i)

    with db() as c:
        for vid, info, err in results:
            r, p, st = found[vid]
            c.execute(
                "INSERT OR REPLACE INTO videos (id, path, root, rel, size, mtime, info, error, present, scanned)"
                " VALUES (?,?,?,?,?,?,?,?,1,?)",
                (vid, p, r["path"], os.path.relpath(p, r["path"]), st.st_size, st.st_mtime,
                 json.dumps(info) if info else None, err, now))
        for vid, (r, p, st) in found.items():
            c.execute("UPDATE videos SET present=1, path=?, root=?, rel=? WHERE id=?",
                      (p, r["path"], os.path.relpath(p, r["path"]), vid))
        missing = 0
        for r in roots:
            rows = c.execute("SELECT id, path FROM videos WHERE root=? AND present=1", (r["path"],)).fetchall()
            for row in rows:
                if row["id"] not in found:
                    c.execute("UPDATE videos SET present=0 WHERE id=?", (row["id"],))
                    missing += 1
    return {"files": len(found), "probed": len(todo),
            "errors": sum(1 for _, _, e in results if e), "missing": missing}


def list_videos(under: list[str] | None = None, include_errors: bool = False) -> list[dict]:
    """Present videos, optionally only those under the given directories."""
    with db() as c:
        rows = [_row(r) for r in c.execute("SELECT * FROM videos WHERE present=1 ORDER BY root, rel")]
    if under:
        rows = [v for v in rows if any(config.inside(v["path"], d) for d in under)]
    if not include_errors:
        rows = [v for v in rows if v["info"]]
    return rows


def get(vid: str) -> dict | None:
    with db() as c:
        r = c.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone()
    return _row(r) if r else None


def require(vid: str) -> dict:
    """A present, probed library video whose path is still inside a known root."""
    v = get(vid)
    if not v or not v["present"] or not v["info"]:
        raise LookupError(f"unknown video {vid}")
    if not os.path.exists(v["path"]):
        raise LookupError(f"video file missing: {v['path']}")
    return v


def mark_missing(vids: list[str]) -> None:
    with db() as c:
        c.executemany("UPDATE videos SET present=0 WHERE id=?", [(v,) for v in vids])


def add_path(path: str, root: str | None = None) -> dict:
    """Index a single file (after an upload or a move), under `root` or
    whichever configured root contains it."""
    path = os.path.abspath(path)
    if root and config.inside(path, root):
        root = {"path": root}
    else:
        root = next((r for r in config.roots() if config.inside(path, r["path"])), None)
    if root is None:
        raise ValueError("file is not inside a library root")
    st = os.stat(path)
    info, err = None, None
    try:
        info = media.probe(path)
    except Exception as e:
        err = str(e)
    vid = video_id(path)
    with db() as c:
        c.execute(
            "INSERT OR REPLACE INTO videos (id, path, root, rel, size, mtime, info, error, present, scanned)"
            " VALUES (?,?,?,?,?,?,?,?,1,?)",
            (vid, path, root["path"], os.path.relpath(path, root["path"]), st.st_size, st.st_mtime,
             json.dumps(info) if info else None, err, time.time()))
    return get(vid)


def cache_key(v: dict, *extra) -> str:
    """Cache validity key: changes whenever the file does."""
    return ":".join(str(x) for x in (v["size"], v["mtime"], *extra))


def work_dir(vid: str, *parts: str) -> str:
    """Per-video working folder for derived files (frames, audio, samples)."""
    return os.path.join(config.WORK, "videos", vid, *parts)
