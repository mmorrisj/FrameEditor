"""Logged file moves: every move the suite makes to a user's video files is
recorded in a batch so it can be undone exactly.

- quarantine(): the duplicate remover's "delete". Files go to
  work/quarantine/<batch>/<root name>/<relative path>, nothing is erased.
- purge(): the only code path that erases files, and only files inside the
  quarantine folder.
- undo(): moves a batch's files back to where they came from.
"""
from __future__ import annotations

import os
import secrets
import shutil
import time

from . import config, library
from .db import db


def new_batch() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def _free(dst: str) -> str:
    """dst, or dst with a numeric suffix if something already occupies it."""
    if not os.path.exists(dst):
        return dst
    stem, ext = os.path.splitext(dst)
    i = 2
    while os.path.exists(f"{stem} ({i}){ext}"):
        i += 1
    return f"{stem} ({i}){ext}"


def move_logged(src: str, dst: str, batch: str, label: str, reason: str = "") -> str:
    dst = _free(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    with db() as c:
        c.execute("INSERT INTO moves (batch, label, reason, src, dst, at) VALUES (?,?,?,?,?,?)",
                  (batch, label, reason, src, dst, time.time()))
    return dst


def quarantine(videos: list[dict], reasons: dict[str, str] | None = None) -> dict:
    """Move library videos into a new quarantine batch."""
    reasons = reasons or {}
    batch = new_batch()
    root_names = {config.norm(r["path"]): r["name"] for r in config.roots()}
    moved, failed = [], {}
    for v in videos:
        rootname = root_names.get(config.norm(v["root"]), os.path.basename(v["root"].rstrip("\\/")) or "other")
        dst = os.path.join(config.QUARANTINE, batch, rootname, v["rel"])
        try:
            move_logged(v["path"], dst, batch, "quarantine", reasons.get(v["id"], ""))
            moved.append(v["id"])
        except OSError as e:
            failed[v["id"]] = str(e)
    library.mark_missing(moved)
    return {"batch": batch, "moved": len(moved), "failed": failed}


def batches(label: str | None = None) -> list[dict]:
    q = ("SELECT batch, label, MIN(at) AS at, COUNT(*) AS files, SUM(restored) AS restored,"
         " SUM(purged) AS purged FROM moves {} GROUP BY batch ORDER BY at DESC")
    with db() as c:
        if label:
            rows = c.execute(q.format("WHERE label=?"), (label,)).fetchall()
        else:
            rows = c.execute(q.format("")).fetchall()
    return [dict(r) for r in rows]


def batch_files(batch: str) -> list[dict]:
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM moves WHERE batch=? ORDER BY id", (batch,))]


def undo(batch: str) -> dict:
    """Move a batch's files back. Skips any whose original spot is now taken."""
    restored, skipped = 0, []
    for m in batch_files(batch):
        if m["restored"] or m["purged"]:
            continue
        if not os.path.exists(m["dst"]):
            skipped.append({"file": m["src"], "why": "no longer at " + m["dst"]})
            continue
        if os.path.exists(m["src"]):
            skipped.append({"file": m["src"], "why": "original location is occupied"})
            continue
        os.makedirs(os.path.dirname(m["src"]), exist_ok=True)
        shutil.move(m["dst"], m["src"])
        with db() as c:
            c.execute("UPDATE moves SET restored=1 WHERE id=?", (m["id"],))
        restored += 1
        library.mark_missing([library.video_id(m["dst"])])
        old = library.get(library.video_id(m["src"]))
        try:
            library.add_path(m["src"], root=old["root"] if old else None)
        except ValueError:
            pass  # restored outside every known root; fine, just not indexed
        _prune_up(os.path.dirname(m["dst"]))
    _prune(os.path.join(config.QUARANTINE, batch))
    return {"restored": restored, "skipped": skipped}


def purge(batch: str) -> dict:
    """Permanently delete a quarantine batch's files (irreversible)."""
    deleted = 0
    for m in batch_files(batch):
        if m["restored"] or m["purged"]:
            continue
        if not config.inside(m["dst"], config.QUARANTINE):
            raise ValueError("refusing to purge a batch with files outside quarantine")
    for m in batch_files(batch):
        if m["restored"] or m["purged"]:
            continue
        if os.path.exists(m["dst"]):
            os.remove(m["dst"])
            deleted += 1
        with db() as c:
            c.execute("UPDATE moves SET purged=1 WHERE id=?", (m["id"],))
    _prune(os.path.join(config.QUARANTINE, batch))
    return {"deleted": deleted}


def _prune_up(d: str) -> None:
    """After undoing a folder-organize move, remove the emptied folders up to
    and including its `_groups` folder (never above it)."""
    parts = os.path.normpath(d).split(os.sep)
    if "_groups" not in parts:
        return
    stop = os.sep.join(parts[:len(parts) - parts[::-1].index("_groups") - 1])
    while config.inside(d, stop) and config.norm(d) != config.norm(stop):
        try:
            os.rmdir(d)
        except OSError:
            return
        d = os.path.dirname(d)


def _prune(d: str) -> None:
    """Remove empty directories under (and including) d."""
    if not os.path.isdir(d):
        return
    for dirpath, _, _ in sorted(os.walk(d), key=lambda w: len(w[0]), reverse=True):
        try:
            os.rmdir(dirpath)
        except OSError:
            pass
