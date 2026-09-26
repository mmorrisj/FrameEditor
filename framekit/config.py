"""Configuration: where work files live and which folders form the library.

Library roots come from, in order:
  1. framekit.json at the repo root (or $FRAMEKIT_CONFIG): {"roots": ["D:/Videos", ...]}
  2. $FRAMEKIT_ROOTS, os.pathsep-separated
  3. the built-in uploads folder (work/uploads), always present

Roots are only ever set from the server side (config file or CLI), never from
the web page, so the browser can't point the app at arbitrary disk locations.
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.abspath(os.environ.get("FRAMEKIT_WORK") or os.path.join(REPO, "work"))
CONFIG_PATH = os.environ.get("FRAMEKIT_CONFIG") or os.path.join(REPO, "framekit.json")

UPLOADS = os.path.join(WORK, "uploads")
QUARANTINE = os.path.join(WORK, "quarantine")
EXPORTS = os.path.join(WORK, "exports")
DB_PATH = os.path.join(WORK, "framekit.db")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".wmv", ".flv",
              ".mpg", ".mpeg", ".ts", ".m2ts", ".3gp"}


def _load() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def inside(path: str, parent: str) -> bool:
    p, d = norm(path), norm(parent)
    return p == d or p.startswith(d.rstrip("\\/") + os.sep)


def roots() -> list[dict]:
    """Library roots as [{name, path}] in preference order (earlier = preferred keeper)."""
    paths = list(_load().get("roots", []))
    env = os.environ.get("FRAMEKIT_ROOTS")
    if env:
        paths += [p for p in env.split(os.pathsep) if p]
    paths.append(UPLOADS)
    out, seen = [], set()
    for p in paths:
        ap = os.path.abspath(p)
        if norm(ap) in seen:
            continue
        seen.add(norm(ap))
        name = "uploads" if norm(ap) == norm(UPLOADS) else (os.path.basename(ap.rstrip("\\/")) or ap)
        out.append({"name": name, "path": ap})
    return out


def add_root(path: str) -> None:
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise ValueError(f"not a directory: {path}")
    cfg = _load()
    rs = cfg.setdefault("roots", [])
    if norm(path) not in {norm(r) for r in rs}:
        rs.append(path)
        _save(cfg)


def remove_root(path: str) -> None:
    cfg = _load()
    cfg["roots"] = [r for r in cfg.get("roots", []) if norm(r) != norm(path)]
    _save(cfg)


def settings(section: str) -> dict:
    """A tool's saved settings from framekit.json (empty if none)."""
    return dict(_load().get(section) or {})


def save_settings(section: str, values: dict) -> None:
    cfg = _load()
    cfg.setdefault(section, {}).update(values)
    _save(cfg)


def ensure_dirs() -> None:
    for d in (WORK, UPLOADS, QUARANTINE, EXPORTS):
        os.makedirs(d, exist_ok=True)
