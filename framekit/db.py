"""SQLite store: the library index, a generic per-video feature cache, and the
file-move log (quarantine and organize moves, used for undo).

A fresh connection per `with db() as c:` block keeps it safe across job
threads; WAL mode lets the web layer read while a job writes.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY, path TEXT UNIQUE, root TEXT, rel TEXT,
    size INTEGER, mtime REAL, info TEXT, error TEXT,
    present INTEGER DEFAULT 1, scanned REAL
);
CREATE TABLE IF NOT EXISTS cache (
    video_id TEXT, kind TEXT, key TEXT, data BLOB,
    PRIMARY KEY (video_id, kind)
);
CREATE TABLE IF NOT EXISTS moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT, batch TEXT, label TEXT, reason TEXT,
    src TEXT, dst TEXT, at REAL,
    restored INTEGER DEFAULT 0, purged INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS moves_batch ON moves(batch);
"""

_ready: set[str] = set()
_init_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    if config.DB_PATH not in _ready:
        with _init_lock:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)
            _ready.add(config.DB_PATH)
    return c


@contextmanager
def db():
    c = _connect()
    try:
        yield c
        c.commit()
    finally:
        c.close()


def cache_get(video_id: str, kind: str, key: str) -> bytes | None:
    with db() as c:
        row = c.execute("SELECT key, data FROM cache WHERE video_id=? AND kind=?",
                        (video_id, kind)).fetchone()
    return row["data"] if row and row["key"] == key else None


def cache_put(video_id: str, kind: str, key: str, data: bytes) -> None:
    with db() as c:
        c.execute("INSERT OR REPLACE INTO cache (video_id, kind, key, data) VALUES (?,?,?,?)",
                  (video_id, kind, key, data))
