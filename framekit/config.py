"""Configuration: where work files live and which folders form the library.

Directories can be set with environment variables, or in a .env file at the
repo root (or $FRAMEKIT_ENV); a real environment variable beats the file.
Relative paths are taken from the repo folder. See .env.example for the list.

Library roots come from, in order:
  1. framekit.json at the repo root (or $FRAMEKIT_CONFIG): {"roots": ["D:/Videos", ...]}
  2. $FRAMEKIT_ROOTS, os.pathsep-separated (";" on Windows)
  3. the built-in uploads folder (work/uploads), always present

Roots are only ever set from the server side (config file, CLI or environment),
never from the web page, so the browser can't point the app at arbitrary disk
locations.
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path: str) -> None:
    """Minimal .env reader (KEY=value, # comments, optional quotes). Variables already
    set in the real environment win."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = (x.strip() for x in line.split("=", 1))
            if key.startswith("export "):
                key = key[7:].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


load_env(os.environ.get("FRAMEKIT_ENV") or os.path.join(REPO, ".env"))


def env_path(name: str, default: str) -> str:
    """Directory from $name (relative to the repo folder), else `default`."""
    v = (os.environ.get(name) or "").strip()
    return os.path.abspath(os.path.join(REPO, os.path.expanduser(v))) if v else os.path.abspath(default)


def env_paths(name: str) -> list[str]:
    """A list of directories from $name, separated by os.pathsep (";" on Windows)."""
    return [os.path.abspath(os.path.join(REPO, os.path.expanduser(p.strip())))
            for p in (os.environ.get(name) or "").split(os.pathsep) if p.strip()]


WORK = env_path("FRAMEKIT_WORK", os.path.join(REPO, "work"))
CONFIG_PATH = env_path("FRAMEKIT_CONFIG", os.path.join(REPO, "framekit.json"))

UPLOADS = env_path("FRAMEKIT_UPLOADS", os.path.join(WORK, "uploads"))
QUARANTINE = env_path("FRAMEKIT_QUARANTINE", os.path.join(WORK, "quarantine"))
EXPORTS = env_path("FRAMEKIT_EXPORTS", os.path.join(WORK, "exports"))
COLORMATCH = env_path("FRAMEKIT_COLORMATCH_DIR", os.path.join(WORK, "colormatch"))
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
    paths = list(_load().get("roots", [])) + env_paths("FRAMEKIT_ROOTS")
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
    for d in (WORK, UPLOADS, QUARANTINE, EXPORTS, COLORMATCH):
        os.makedirs(d, exist_ok=True)


def own_dirs() -> set[str]:
    """FrameKit's output folders (normalised), which library scans must never index,
    wherever they have been moved to. Uploads are the exception: they are a root."""
    return {norm(d) for d in (WORK, QUARANTINE, EXPORTS, COLORMATCH)} - {norm(UPLOADS)}


def directories() -> dict:
    """Every configured directory, for `framekit dirs` and the startup log."""
    return {"work": WORK, "config": CONFIG_PATH, "uploads": UPLOADS, "exports": EXPORTS,
            "quarantine": QUARANTINE, "colormatch": COLORMATCH,
            "library roots (from FRAMEKIT_ROOTS)": env_paths("FRAMEKIT_ROOTS"),
            "lineage folders (from FRAMEKIT_LINEAGE_DIRS)": env_paths("FRAMEKIT_LINEAGE_DIRS"),
            "frame inbox (FRAMEKIT_FRAME_INBOX)": os.environ.get("FRAMEKIT_FRAME_INBOX") or "(default)"}
