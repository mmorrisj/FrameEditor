"""Lineage: work out which AI clip continues which, from first and last frames.

A clip generated from another clip's last frame starts on (a copy of) that
frame, so parents and children can be matched without trusting file names:

  takes       clips whose first frames match (same start image) form one step
  parent      a step's first frame matches the last frame of a clip outside it;
              that clip is the step's parent, and the take that was continued
  root        a step with no parent starts a chain at generation 1

The forest is rebuilt from the index every time, so arrival order doesn't
matter (a child indexed before its parent links up once the parent appears),
and manual link/unlink overrides are applied on top. Chain numbers are stored,
so labels like c007_g03b_t2 (chain 7, generation 3, second branch, take 2)
never shift. The rules follow the LastFrame lineage tool; frames here are
decoded by ffmpeg with the source's own color matrix.

Folders are indexed non-recursively (so stitched outputs in a subfolder can't
pose as takes). A view can span every saved folder or just one.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from .. import config, media
from ..db import db
from .colormatch import color_in

SIG = 64                 # signatures are SIG x SIG grayscale, z-scored
THUMB_H = 240            # preview JPEG height
MATCH = 0.99             # correlation above which two frames are "the same"
SUGGEST = 0.90           # near-miss parent shown for a root
SETTLE = 5.0             # seconds a file must be untouched before it's indexed
_NOWIN = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}

SCHEMA = """
CREATE TABLE IF NOT EXISTS lineage_clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, folder TEXT, name TEXT,
    size INTEGER, mtime_ns INTEGER, width INTEGER, height INTEGER, frames INTEGER,
    fps REAL, fps_frac TEXT, duration REAL, has_audio INTEGER,
    first_sig BLOB, last_sig BLOB, missing INTEGER DEFAULT 0, added REAL
);
CREATE INDEX IF NOT EXISTS lineage_clips_folder ON lineage_clips(folder);
CREATE TABLE IF NOT EXISTS lineage_overrides (
    child_id INTEGER, parent_id INTEGER, kind TEXT, PRIMARY KEY (child_id, parent_id)
);
CREATE TABLE IF NOT EXISTS lineage_chains (root_key INTEGER PRIMARY KEY, number INTEGER UNIQUE);
CREATE TABLE IF NOT EXISTS lineage_picks (root_key INTEGER PRIMARY KEY, tip INTEGER);
CREATE TABLE IF NOT EXISTS lineage_marks (clip_id INTEGER PRIMARY KEY, mark TEXT);
"""
_schema_ok: set[str] = set()
_lock = threading.Lock()
_version = 0            # bumped on every index or override change; invalidates cached forests
_forests: dict = {}


def _db():
    if config.DB_PATH not in _schema_ok:
        with db() as c:
            c.executescript(SCHEMA)
        _schema_ok.add(config.DB_PATH)
    return db()


def _bump():
    global _version
    with _lock:
        _version += 1
        _forests.clear()


def thumb_dir() -> str:
    return os.path.join(config.WORK, "lineage", "thumbs")


def thumb_path(cid: int, which: str) -> str:
    return os.path.join(thumb_dir(), f"{int(cid)}_{which}.jpg")


# ==== folders ======================================================================================

def folders() -> list[str]:
    return list(config.settings("lineage").get("folders") or [])


def add_folder(path: str) -> list[str]:
    path = os.path.abspath(path.strip().strip('"'))
    if not os.path.isdir(path):
        raise ValueError(f"no such folder: {path}")
    fs = folders()
    if not any(config.norm(f) == config.norm(path) for f in fs):
        fs.append(path)
        config.save_settings("lineage", {"folders": fs})
    _bump()
    return fs


def remove_folder(path: str) -> list[str]:
    fs = [f for f in folders() if config.norm(f) != config.norm(path)]
    config.save_settings("lineage", {"folders": fs})
    _bump()
    return fs


def _videos_in(folder: str) -> list[str]:
    try:
        names = os.listdir(folder)
    except OSError as e:
        raise ValueError(f"can't read {folder}: {e}")
    return sorted(os.path.join(folder, n) for n in names
                  if os.path.splitext(n)[1].lower() in config.VIDEO_EXTS
                  and os.path.isfile(os.path.join(folder, n)))


# ==== probing =======================================================================================

def signature(img: np.ndarray) -> np.ndarray:
    g = Image.fromarray(img).convert("L").resize((SIG, SIG), Image.BOX)
    s = np.asarray(g, np.float32).ravel()
    s -= s.mean()
    s /= s.std() + 1e-6
    return s


def _decode(path: str, info: dict, tail: bool) -> np.ndarray | None:
    """First (or last) frame at preview size, color-exact. The last frame is
    the final decodable one, which skips broken trailing frames."""
    w, h = info["width"], info["height"]
    th = THUMB_H if h >= THUMB_H else h - h % 2
    tw = max(2, int(round(w * th / h / 2)) * 2)
    vf = f"scale={tw}:{th}:{color_in(info)}:flags=area+accurate_rnd+full_chroma_int,format=rgb24"
    size = tw * th * 3
    tries = ([["-sseof", "-1.5"], []] if tail else [[]])
    for pre in tries:
        cmd = ["ffmpeg", "-v", "error", *pre, "-i", path, "-an", "-vf", vf]
        cmd += ([] if tail else ["-frames:v", "1"]) + ["-f", "rawvideo", "-"]
        proc = subprocess.run(cmd, capture_output=True, **_NOWIN)
        n = len(proc.stdout) // size
        if n:
            k = n - 1 if tail else 0
            return np.frombuffer(proc.stdout[k * size:(k + 1) * size], np.uint8).reshape(th, tw, 3)
    return None


def probe_clip(path: str) -> dict:
    info = media.probe(path)
    first = _decode(path, info, tail=False)
    last = _decode(path, info, tail=True)
    if first is None or last is None:
        raise RuntimeError("no decodable frames")
    frames = info.get("nb_frames") or int(round((info.get("duration") or 0) * (info.get("fps") or 0)))
    return {"info": info, "frames": frames, "first": first, "last": last}


def scan(job=None, only: list[str] | None = None) -> dict:
    """Index new or changed clips in the saved folders (or just `only`), follow
    moved/renamed files, and flag vanished ones."""
    targets = [os.path.abspath(f) for f in (only or folders())]
    if not targets:
        raise ValueError("add a folder first")
    todo, seen, moved, waiting = [], set(), 0, 0
    now = time.time()
    with _db() as c:
        for folder in targets:
            for p in _videos_in(folder):
                seen.add(config.norm(p))
                st = os.stat(p)
                row = c.execute("SELECT id, size, mtime_ns, missing FROM lineage_clips WHERE path=?", (p,)).fetchone()
                if row and row["size"] == st.st_size and row["mtime_ns"] == st.st_mtime_ns:
                    if row["missing"]:
                        c.execute("UPDATE lineage_clips SET missing=0 WHERE id=?", (row["id"],))
                    continue
                if now - st.st_mtime < SETTLE:
                    waiting += 1   # still being written or synced; next scan picks it up
                    continue
                if not row:  # the same file under a new name or folder?
                    gone = [r for r in c.execute(
                        "SELECT id, path FROM lineage_clips WHERE size=? AND mtime_ns=?",
                        (st.st_size, st.st_mtime_ns)) if not os.path.exists(r["path"])]
                    if gone:
                        c.execute("UPDATE lineage_clips SET path=?, folder=?, name=?, missing=0 WHERE id=?",
                                  (p, folder, os.path.basename(p), gone[0]["id"]))
                        moved += 1
                        continue
                todo.append((p, folder, st))
    # oldest first, so a backfill gets ids (and take numbers) in creation order
    todo.sort(key=lambda t: t[2].st_mtime_ns)
    os.makedirs(thumb_dir(), exist_ok=True)
    errors = []
    if job:
        job.update(message="reading first and last frames", progress=0, total=len(todo))
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:  # clips on a synced drive are I/O bound
        futs = [(t, pool.submit(probe_clip, t[0])) for t in todo]
        for (p, folder, st), fut in futs:
            if job:
                job.check()
            try:
                pr = fut.result()
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
                continue
            info = pr["info"]
            vals = (folder, os.path.basename(p), st.st_size, st.st_mtime_ns, info["width"], info["height"],
                    pr["frames"], info.get("fps") or 0, info.get("fps_frac"), info.get("duration") or 0,
                    int(info.get("has_audio") or 0), signature(pr["first"]).tobytes(),
                    signature(pr["last"]).tobytes())
            with _db() as c:
                row = c.execute("SELECT id FROM lineage_clips WHERE path=?", (p,)).fetchone()
                if row:
                    cid = row["id"]
                    c.execute("UPDATE lineage_clips SET folder=?, name=?, size=?, mtime_ns=?, width=?, height=?,"
                              " frames=?, fps=?, fps_frac=?, duration=?, has_audio=?, first_sig=?, last_sig=?,"
                              " missing=0 WHERE id=?", vals + (cid,))
                else:
                    cid = c.execute("INSERT INTO lineage_clips (folder, name, size, mtime_ns, width, height, frames,"
                                    " fps, fps_frac, duration, has_audio, first_sig, last_sig, path, added)"
                                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                    vals + (p, time.time())).lastrowid
            Image.fromarray(pr["first"]).save(thumb_path(cid, "first"), quality=82)
            Image.fromarray(pr["last"]).save(thumb_path(cid, "last"), quality=82)
            done += 1
            if job:
                job.update(progress=done)
    gone = 0
    with _db() as c:
        for folder in targets:
            for r in c.execute("SELECT id, path, missing FROM lineage_clips WHERE folder=?", (folder,)).fetchall():
                miss = int(config.norm(r["path"]) not in seen)
                if miss != r["missing"]:
                    c.execute("UPDATE lineage_clips SET missing=? WHERE id=?", (miss, r["id"]))
                    gone += miss
    _bump()
    return {"probed": done, "moved": moved, "missing": gone, "waiting": waiting, "errors": errors[:20],
            "folders": targets}


# ==== the forest =====================================================================================

class Group:
    __slots__ = ("takes", "dups", "parent", "link", "alts", "suggestion", "children", "chain",
                 "generation", "label", "up")

    def __init__(self, takes):
        self.takes = takes          # clip ids, arrival order
        self.dups = []              # (duplicate id, canonical id)
        self.parent = None          # clip id
        self.link = ""              # exact | probable (resolution differs) | manual
        self.alts = []              # other clips that also matched as parent
        self.suggestion = None      # (clip id, score) near-miss parent for a root
        self.children = []
        self.chain = 0
        self.generation = 1
        self.label = ""
        self.up = None              # parent Group

    @property
    def key(self):
        return min(self.takes + [d for d, _ in self.dups])


def _clips(scope: str | None) -> list[dict]:
    fs = [os.path.abspath(scope)] if scope else [os.path.abspath(f) for f in folders()]
    if not fs:
        return []
    q = ",".join("?" * len(fs))
    with _db() as c:
        rows = c.execute(f"SELECT * FROM lineage_clips WHERE missing=0 AND folder IN ({q}) ORDER BY id", fs).fetchall()
    return [dict(r) for r in rows]


def forest(scope: str | None = None) -> dict:
    """The chains for every saved folder (scope None) or one folder, cached until something changes."""
    key = (config.DB_PATH, scope and config.norm(scope))
    with _lock:
        hit = _forests.get(key)
        if hit and hit["version"] == _version:
            return hit
        ver = _version
    f = _build(_clips(scope))
    f["version"] = ver
    with _lock:
        _forests[key] = f
    return f


def _build(clips: list[dict]) -> dict:
    n = len(clips)
    out = {"clips": {c["id"]: c for c in clips}, "groups": [], "group_of": {}, "dup_of": {},
           "labels": {}, "roots": [], "winners": set()}
    if not n:
        return out
    ids = [c["id"] for c in clips]
    pos = {cid: i for i, cid in enumerate(ids)}
    F = np.stack([np.frombuffer(c["first_sig"], np.float32) for c in clips])
    L = np.stack([np.frombuffer(c["last_sig"], np.float32) for c in clips])
    d = F.shape[1]
    res = np.array([(c["width"], c["height"]) for c in clips])
    same_res = (res[:, None, :] == res[None, :, :]).all(-1)
    frames = np.array([c["frames"] for c in clips])
    first_eq = (F @ F.T / d > MATCH) & same_res
    last_eq = (L @ L.T / d > MATCH) & same_res
    FL = F @ L.T / d            # FL[i, j]: first frame of i vs last frame of j

    # duplicates: same first frame, last frame and length -> one clip saved twice
    dup = first_eq & last_eq & (frames[:, None] == frames[None, :])
    canon = np.arange(n)
    for i in range(n):
        if canon[i] == i:
            for j in np.nonzero(dup[i, i + 1:])[0] + i + 1:
                if canon[j] == j:
                    canon[j] = i
    keep = [i for i in range(n) if canon[i] == i]

    # takes: union-find over canonical clips sharing a first frame
    uf = list(range(n))

    def find(i):
        while uf[i] != i:
            uf[i] = uf[uf[i]]
            i = uf[i]
        return i
    for i in keep:
        for j in np.nonzero(first_eq[i, i + 1:])[0] + i + 1:
            if canon[j] == j:
                uf[find(j)] = find(i)
    members: dict[int, list[int]] = {}
    for i in keep:
        members.setdefault(find(i), []).append(i)
    groups, at = [], {}
    for idxs in members.values():
        g = Group([ids[i] for i in idxs])
        groups.append(g)
        for i in idxs:
            at[i] = g
    for j in range(n):
        if canon[j] != j:
            at[canon[j]].dups.append((ids[j], ids[canon[j]]))
            out["dup_of"][ids[j]] = ids[canon[j]]

    with _db() as c:
        ovr = c.execute("SELECT child_id, parent_id, kind FROM lineage_overrides").fetchall()
        numbers = {r["root_key"]: r["number"] for r in c.execute("SELECT * FROM lineage_chains")}
    forced, unlinked = {}, set()
    for r in ovr:
        if r["child_id"] in pos and r["parent_id"] in pos:
            g, p = at[canon[pos[r["child_id"]]]], canon[pos[r["parent_id"]]]
            if r["kind"] == "link":
                forced[g] = p
            else:
                unlinked.add((g, p))

    for g in groups:
        rep = pos[g.takes[0]]
        own = {pos[t] for t in g.takes}
        if g in forced and forced[g] not in own:
            g.parent, g.link = ids[forced[g]], "manual"
            continue
        hits = [j for j in np.nonzero(FL[rep] > MATCH)[0]
                if canon[j] == j and j not in own and (g, j) not in unlinked]
        if hits:  # same resolution first, then the best score, then the oldest
            hits.sort(key=lambda j: (not same_res[rep, j], -FL[rep, j], j))
            g.parent = ids[hits[0]]
            g.link = "exact" if same_res[rep, hits[0]] else "probable"
            g.alts = [ids[j] for j in hits[1:]]

    # break cycles at the step that appeared first
    for g in groups:
        seen, cur = [], g
        while cur.parent is not None and cur not in seen:
            seen.append(cur)
            cur = at[pos[cur.parent]]
        if cur.parent is not None:
            origin = min(seen[seen.index(cur):], key=lambda x: x.takes[0])
            origin.parent, origin.link, origin.alts = None, "", []

    for g in groups:
        if g.parent is not None:
            g.up = at[pos[g.parent]]
            g.up.children.append(g)

    for g in groups:  # near-miss parents for roots (e.g. an edited handoff frame)
        if g.parent is None:
            rep = pos[g.takes[0]]
            own = {pos[t] for t in g.takes}
            best = [(float(FL[rep, j]), ids[j]) for j in keep if j not in own and FL[rep, j] > SUGGEST]
            if best:
                s, cid = max(best)
                g.suggestion = (cid, s)

    roots = sorted((g for g in groups if g.parent is None), key=lambda g: g.takes[0])
    new = {}
    for root in roots:
        if root.key not in numbers:
            numbers[root.key] = new[root.key] = max(numbers.values(), default=0) + 1
        todo = [(root, 1)]
        while todo:
            g, gen = todo.pop()
            g.chain, g.generation = numbers[root.key], gen
            g.children.sort(key=lambda x: x.takes[0])
            todo.extend((ch, gen + 1) for ch in g.children)
    if new:
        with _db() as c:
            c.executemany("INSERT OR IGNORE INTO lineage_chains VALUES (?, ?)", new.items())

    siblings: dict = {}
    for g in sorted(groups, key=lambda g: g.takes[0]):
        k = siblings.setdefault((g.chain, g.generation), [])
        suffix = "" if not k else (chr(ord("a") + len(k)) if len(k) < 26 else f"x{len(k)}")
        g.label = f"c{g.chain:03d}_g{g.generation:02d}{suffix}"
        k.append(g)
    for g in groups:
        for t, cid in enumerate(g.takes, 1):
            out["labels"][cid] = f"{g.label}_t{t}"
            out["group_of"][cid] = g
    count: dict = {}
    for dup_id, cid in sorted(out["dup_of"].items()):
        count[cid] = count.get(cid, 0) + 1
        out["labels"][dup_id] = f"{out['labels'][cid]}_dup{count[cid]}"
    out["groups"] = groups
    out["roots"] = sorted(roots, key=lambda g: g.chain)
    out["winners"] = {g.parent for g in groups if g.parent is not None}
    return out


def _walk(root: Group) -> list[Group]:
    out, todo = [], [root]
    while todo:
        g = todo.pop(0)
        out.append(g)
        todo.extend(g.children)
    return out


def _depth(g: Group) -> int:
    return 1 + max((_depth(c) for c in g.children), default=0)


def path_to(f: dict, cid: int) -> list[int]:
    """Clip ids from the chain's root down to `cid`, in playback order."""
    cid = f["dup_of"].get(cid, cid)
    path, g = [cid], f["group_of"][cid]
    while g.parent is not None:
        path.append(g.parent)
        g = f["group_of"][g.parent]
    return path[::-1]


def _marks() -> dict[int, str]:
    with _db() as c:
        return {r["clip_id"]: r["mark"] for r in c.execute("SELECT * FROM lineage_marks")}


def default_tip(f: dict, root: Group, marks: dict) -> int:
    """Follow the continued takes down the deepest branch; end on the newest take not rejected."""
    g = root
    while g.children:
        g = max(g.children, key=lambda c: (_depth(c), c.takes[0]))
    ok = [t for t in g.takes if marks.get(t) != "reject"] or g.takes
    return ok[-1]


def chains(scope: str | None = None) -> list[dict]:
    f = forest(scope)
    out = []
    for root in f["roots"]:
        gs = _walk(root)
        cl = [t for g in gs for t in g.takes]
        out.append({"number": root.chain, "label": f"c{root.chain:03d}", "key": root.key, "first": root.takes[0],
                    "depth": _depth(root), "clips": len(cl), "steps": len(gs),
                    "updated": max(f["clips"][t]["mtime_ns"] for t in cl) / 1e9,
                    "names": [f["clips"][t]["name"] for t in cl][:3],
                    "suggestion": bool(root.suggestion)})
    out.sort(key=lambda c: -c["updated"])
    return out


def _root_of(f: dict, number: int) -> Group:
    root = next((r for r in f["roots"] if r.chain == int(number)), None)
    if root is None:
        raise LookupError(f"no chain c{int(number):03d} here")
    return root


def _clip_public(f: dict, cid: int, marks: dict) -> dict:
    c = f["clips"][cid]
    return {"id": cid, "name": c["name"], "label": f["labels"].get(cid), "width": c["width"],
            "height": c["height"], "frames": c["frames"], "fps": c["fps"], "duration": c["duration"],
            "has_audio": bool(c["has_audio"]), "mark": marks.get(cid), "continued": cid in f["winners"],
            "mtime": c["mtime_ns"] / 1e9}


def chain(number: int, scope: str | None = None) -> dict:
    """One chain in full: its steps generation by generation, and the selected path."""
    f = forest(scope)
    root = _root_of(f, number)
    marks = _marks()
    gs = _walk(root)
    clips = {}
    steps = []
    for g in gs:
        for t in g.takes:
            clips[t] = _clip_public(f, t, marks)
        for dup, _ in g.dups:
            clips[dup] = _clip_public(f, dup, marks)
        sug = None
        if g.suggestion:
            sid, score = g.suggestion
            sug = {"clip": sid, "label": f["labels"].get(sid), "name": f["clips"][sid]["name"], "score": round(score, 3)}
        steps.append({"label": g.label, "generation": g.generation, "takes": g.takes,
                      "dups": [d for d, _ in g.dups], "parent": g.parent, "link": g.link,
                      "alternates": [{"clip": a, "label": f["labels"].get(a)} for a in g.alts],
                      "suggestion": sug, "children": [c.label for c in g.children],
                      "up": g.up.label if g.up else None})
    with _db() as c:
        row = c.execute("SELECT tip FROM lineage_picks WHERE root_key=?", (root.key,)).fetchone()
    tip, picked = None, False
    if row and row["tip"] in f["group_of"] and f["group_of"][row["tip"]].chain == root.chain:
        tip, picked = row["tip"], True
    if tip is None:
        tip = default_tip(f, root, marks)
    return {"number": root.chain, "label": f"c{root.chain:03d}", "key": root.key, "depth": _depth(root),
            "steps": steps, "clips": clips, "path": path_to(f, tip), "picked": picked}


def pick(number: int, tip: int | None, scope: str | None = None) -> dict:
    """Select the chain that ends at clip `tip` (None: back to the automatic choice)."""
    f = forest(scope)
    root = _root_of(f, number)
    with _db() as c:
        if tip is None:
            c.execute("DELETE FROM lineage_picks WHERE root_key=?", (root.key,))
        else:
            tip = f["dup_of"].get(int(tip), int(tip))
            if tip not in f["group_of"] or f["group_of"][tip].chain != root.chain:
                raise ValueError("that clip isn't in this chain")
            c.execute("INSERT OR REPLACE INTO lineage_picks VALUES (?, ?)", (root.key, tip))
    return chain(number, scope)


def mark(cid: int, value: str | None) -> None:
    if value not in (None, "", "star", "reject"):
        raise ValueError("mark must be star, reject or empty")
    with _db() as c:
        if value:
            c.execute("INSERT OR REPLACE INTO lineage_marks VALUES (?, ?)", (int(cid), value))
        else:
            c.execute("DELETE FROM lineage_marks WHERE clip_id=?", (int(cid),))


def override(child: int, parent: int | None, kind: str) -> None:
    """kind link: force child's step onto parent; unlink: reject that parent; reset: drop both."""
    with _db() as c:
        if not c.execute("SELECT 1 FROM lineage_clips WHERE id=?", (int(child),)).fetchone():
            raise LookupError("unknown clip")
        if kind == "reset":
            c.execute("DELETE FROM lineage_overrides WHERE child_id=?", (int(child),))
        elif kind in ("link", "unlink"):
            if parent is None or not c.execute("SELECT 1 FROM lineage_clips WHERE id=?", (int(parent),)).fetchone():
                raise LookupError("unknown parent clip")
            if int(parent) == int(child):
                raise ValueError("a clip can't be its own parent")
            if kind == "link":  # a step has one parent: a new forced link replaces an older one
                c.execute("DELETE FROM lineage_overrides WHERE child_id=? AND kind='link'", (int(child),))
            c.execute("INSERT OR REPLACE INTO lineage_overrides VALUES (?, ?, ?)", (int(child), int(parent), kind))
        else:
            raise ValueError("kind must be link, unlink or reset")
    _bump()


def resolve(text: str, scope: str | None = None) -> int | None:
    """A clip id from a label (c007_g03_t2), step (c007_g03 -> its first take), or file name."""
    f = forest(scope)
    t = text.strip().lower()
    for cid, lab in f["labels"].items():
        if lab.lower() == t:
            return cid
    for g in f["groups"]:
        if g.label.lower() == t:
            return g.takes[0]
    for cid, c in f["clips"].items():
        if t in (c["name"].lower(), os.path.splitext(c["name"])[0].lower()):
            return cid
    return None


def clip(cid: int) -> dict:
    with _db() as c:
        r = c.execute("SELECT * FROM lineage_clips WHERE id=?", (int(cid),)).fetchone()
    if not r:
        raise LookupError("unknown clip")
    return dict(r)


def suggest(cid: int, scope: str | None = None, backend: str = "clip", top: int = 6) -> list[dict]:
    """Likeliest parents for clip `cid`'s step by content, for handoff frames that were
    edited (restyled, retouched) before reuse, where the exact match can't see them."""
    from .. import features
    f = forest(scope)
    cid = f["dup_of"].get(int(cid), int(cid))
    if cid not in f["group_of"]:
        raise LookupError("unknown clip")
    g = f["group_of"][cid]
    own = set(g.takes)
    below = set()
    todo = [g]
    while todo:  # a step's own descendants can't be its parent
        x = todo.pop()
        below.update(x.takes)
        todo.extend(x.children)
    cands = [c for c in f["clips"] if c not in own and c not in below and c not in f["dup_of"]]
    if not cands:
        return []
    if backend == "clip" and not features.backends().get("clip"):
        backend = "visual"
    emb = _embeddings([(cid, "first")] + [(c, "last") for c in cands], backend, f)
    sims = emb[1:] @ emb[0]
    order = np.argsort(-sims)[:top]
    return [{"clip": cands[i], "label": f["labels"].get(cands[i]), "name": f["clips"][cands[i]]["name"],
             "score": round(float(sims[i]), 3), "backend": backend} for i in order]


def _embeddings(items: list[tuple[int, str]], backend: str, f: dict) -> np.ndarray:
    """Unit-length embeddings of clip thumbnails, cached per clip version in the feature cache."""
    from .. import features
    from ..db import cache_get, cache_put
    out: list = [None] * len(items)
    todo = []
    for k, (cid, which) in enumerate(items):
        c = f["clips"][cid]
        key = f"{c['size']}:{c['mtime_ns']}"
        data = cache_get(f"lineage:{cid}", f"emb-{backend}-{which}", key)
        if data:
            out[k] = np.frombuffer(data, np.float32)
        else:
            todo.append((k, cid, which, key))
    for s in range(0, len(todo), 64):
        chunk = todo[s:s + 64]
        imgs = []
        for _, cid, which, _ in chunk:
            with Image.open(thumb_path(cid, which)) as im:
                imgs.append(im.convert("RGB"))
        emb = features.embed(imgs, backend).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        for (k, cid, which, key), e in zip(chunk, emb):
            out[k] = e
            cache_put(f"lineage:{cid}", f"emb-{backend}-{which}", key, e.tobytes())
    return np.stack(out)


def tree_lines(scope: str | None = None, number: int | None = None) -> list[str]:
    """Terminal view of the chains."""
    f = forest(scope)
    lines = []
    for root in f["roots"]:
        if number is not None and root.chain != number:
            continue
        gs = _walk(root)
        lines.append(f"c{root.chain:03d}  {_depth(root)} generation(s), {sum(len(g.takes) for g in gs)} take(s)")
        todo = [(root, 1)]
        while todo:
            g, ind = todo.pop()
            marks = " ".join(f"t{n}{'*' if t in f['winners'] else ''}" for n, t in enumerate(g.takes, 1))
            if g.parent is None:
                note = "new chain"
                if g.suggestion:
                    note += f" (maybe from {f['labels'][g.suggestion[0]]}, score {g.suggestion[1]:.3f})"
            else:
                note = f"from {f['labels'][g.parent]}" + (f" [{g.link}]" if g.link != "exact" else "")
            dups = f" (+{len(g.dups)} dup)" if g.dups else ""
            lines.append(f"{'  ' * ind}{g.label}  [{marks}]{dups}  {note}")
            todo.extend((ch, ind + 1) for ch in reversed(g.children))
        lines.append("")
    return lines
