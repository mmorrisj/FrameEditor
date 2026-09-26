"""Duplicate video detection and removal.

Tiers, cheapest first:
  exact      identical bytes: grouped by size, confirmed by SHA-256.
  near       the same video re-encoded, rescaled or remuxed: durations match
             within tolerance and the perceptual hashes of frames sampled at
             the same relative positions are close.
  contained  (deep scan) a trimmed copy: the shorter video's 1 fps hash
             sequence lines up inside the longer one. Reported separately and
             never auto-selected for removal, since the trim may be the keeper.
  audio      (optional, needs Chromaprint's fpcalc) same-length videos whose
             audio fingerprints match. Catches crops and overlays that defeat
             frame hashes.

Removal never deletes: extras move to quarantine (see fileops), undoable.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time
from collections import defaultdict

import numpy as np

from .. import config, fileops, library, samples
from ..db import cache_get, cache_put
from ..features import hamming

DEFAULTS = {
    "near_threshold": 10,      # mean Hamming bits (of 64) across sampled frames
    "duration_tol": 0.02,      # fraction of the longer duration...
    "duration_min_tol": 1.0,   # ...but at least this many seconds
    "contain_threshold": 12,   # mean Hamming bits along the best alignment
    "min_contain_s": 3.0,      # shorter clip must be at least this long
    "audio_threshold": 0.15,   # Chromaprint bit error rate
}

REPORT_DIR = lambda: os.path.join(config.WORK, "dupes")


# --- helpers -------------------------------------------------------------------

class _UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def sha256(v: dict) -> str:
    key = library.cache_key(v)
    raw = cache_get(v["id"], "sha256", key)
    if raw:
        return raw.decode()
    h = hashlib.sha256()
    with open(v["path"], "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    cache_put(v["id"], "sha256", key, digest.encode())
    return digest


def _dur(v: dict) -> float:
    return float(v["info"].get("duration") or 0)


def _tol(a: float, b: float, p: dict) -> float:
    return max(p["duration_min_tol"], p["duration_tol"] * max(a, b))


def _bitrate(v: dict) -> float:
    d = _dur(v)
    return v["info"].get("bitrate") or (v["size"] * 8 / d if d else 0)


def keeper_rank(v: dict, root_pref: dict[str, int]):
    """Sort key: highest resolution, bitrate, duration; then preferred root, oldest."""
    i = v["info"]
    return (-(i["width"] * i["height"]), -_bitrate(v), -_dur(v),
            root_pref.get(config.norm(v["root"]), 99), v["mtime"])


def align(short_h, short_v, long_h, long_v) -> tuple[float, int]:
    """Best (mean Hamming, offset) of the short sequence slid along the long one.

    Sequences are sampled at 1 fps with arbitrary phase, so each frame may
    match either of the two nearest frames of the other sequence.
    """
    ls, ll = len(short_h), len(long_h)
    if ls == 0 or ll < ls:
        return 64.0, 0
    D = hamming(short_h[:, None], long_h[None, :]).astype(np.float32)
    D = np.minimum(D, np.concatenate([D[:, 1:], np.full((ls, 1), 64, np.float32)], axis=1))
    D[~short_v, :] = np.nan
    D[:, ~long_v] = np.nan
    rows = np.arange(ls)
    best, best_o = 64.0, 0
    for o in range(ll - ls + 1):
        d = D[rows, rows + o]
        ok = ~np.isnan(d)
        if ok.sum() < max(2, ls // 2):
            continue
        m = float(d[ok].mean())
        if m < best:
            best, best_o = m, o
    return best, best_o


def _fpcalc(path: str) -> np.ndarray | None:
    exe = shutil.which("fpcalc")
    if not exe:
        return None
    proc = subprocess.run([exe, "-raw", "-length", "120", path], capture_output=True, text=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in proc.stdout.splitlines():
        if line.startswith("FINGERPRINT="):
            vals = [int(x) for x in line.split("=", 1)[1].split(",") if x]
            return np.array(vals, dtype=np.int64).astype(np.uint32)
    return None


def _audio_ber(a: np.ndarray, b: np.ndarray, max_shift: int = 40) -> float:
    """Best bit error rate between two raw Chromaprint fingerprints."""
    best = 1.0
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for s in range(-max_shift, max_shift + 1):
        x, y = (a[s:], b) if s >= 0 else (a, b[-s:])
        n = min(len(x), len(y))
        if n < 20:
            continue
        xor = np.bitwise_xor(x[:n], y[:n])
        bits = pop[xor.view(np.uint8)].sum()
        best = min(best, bits / (32.0 * n))
    return best


def audio_available() -> bool:
    return shutil.which("fpcalc") is not None


# --- scan ------------------------------------------------------------------------

def scan(job, under: list[str] | None = None, video_ids: list[str] | None = None,
         deep: bool = False, audio: bool = False, **params) -> dict:
    p = {**DEFAULTS, **{k: v for k, v in params.items() if v is not None}}
    videos = library.list_videos(under)
    if video_ids:
        wanted = set(video_ids)
        videos = [v for v in videos if v["id"] in wanted]
    by_id = {v["id"]: v for v in videos}
    uf = _UF()
    evidence: dict[frozenset, str] = {}
    errors: dict[str, str] = {}

    # 1. exact copies
    by_size = defaultdict(list)
    for v in videos:
        if v["size"]:
            by_size[v["size"]].append(v)
    cands = [v for g in by_size.values() if len(g) > 1 for v in g]
    job.update(message="hashing same-size files", progress=0, total=len(cands))
    shas: dict[str, str] = {}
    for i, v in enumerate(cands, 1):
        job.check()
        try:
            shas[v["id"]] = sha256(v)
        except OSError as e:
            errors[v["id"]] = str(e)
        job.update(progress=i)
    by_sha = defaultdict(list)
    for vid, h in shas.items():
        by_sha[h].append(vid)
    for ids in by_sha.values():
        for other in ids[1:]:
            uf.union(ids[0], other)
            evidence[frozenset((ids[0], other))] = "identical bytes"

    # 2. re-encodes: same duration, similar sampled frames
    got = samples.ensure(videos, job)
    S, errors = got["samples"], {**errors, **got["errors"]}
    order = sorted((v for v in videos if v["id"] in S), key=_dur)
    job.update(message="comparing sampled frames", progress=0, total=len(order))
    for i, a in enumerate(order):
        job.check()
        sa = S[a["id"]]
        for b in order[i + 1:]:
            if _dur(b) - _dur(a) > _tol(_dur(a), _dur(b), p):
                break
            if uf.find(a["id"]) == uf.find(b["id"]):
                continue
            sb = S[b["id"]]
            if len(sa["hashes"]) != len(sb["hashes"]):
                continue
            m = sa["valid"] & sb["valid"]
            if m.sum() < max(3, len(m) // 2):
                continue
            d = float(hamming(sa["hashes"][m], sb["hashes"][m]).mean())
            if d <= p["near_threshold"]:
                uf.union(a["id"], b["id"])
                evidence[frozenset((a["id"], b["id"]))] = f"frames match (avg {d:.1f}/64 bits apart)"
        job.update(progress=i + 1)

    # 3. audio fingerprints (optional)
    audio_note = None
    if audio:
        if not audio_available():
            audio_note = "fpcalc (Chromaprint) not found on PATH; audio check skipped"
        else:
            fps_ = {}
            withaudio = [v for v in order if v["info"].get("has_audio")]
            job.update(message="fingerprinting audio", progress=0, total=len(withaudio))
            for i, v in enumerate(withaudio, 1):
                job.check()
                key = library.cache_key(v, "fp1")
                raw = cache_get(v["id"], "chromaprint", key)
                if raw is None:
                    fp = _fpcalc(v["path"])
                    raw = fp.tobytes() if fp is not None else b""
                    cache_put(v["id"], "chromaprint", key, raw)
                if raw:
                    fps_[v["id"]] = np.frombuffer(raw, dtype=np.uint32)
                job.update(progress=i)
            ids = [v for v in withaudio if v["id"] in fps_]
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    if _dur(b) - _dur(a) > _tol(_dur(a), _dur(b), p):
                        break
                    if uf.find(a["id"]) == uf.find(b["id"]):
                        continue
                    ber = _audio_ber(fps_[a["id"]], fps_[b["id"]])
                    if ber <= p["audio_threshold"]:
                        uf.union(a["id"], b["id"])
                        evidence[frozenset((a["id"], b["id"]))] = f"audio matches ({ber:.0%} bit error)"

    # 4. trims / overlaps (deep)
    contained = []
    if deep:
        seq = {}
        job.update(message="decoding 1 fps sequences", progress=0, total=len(order))
        for i, v in enumerate(order, 1):
            job.check()
            try:
                seq[v["id"]] = samples.sequence(v, job)
            except Exception as e:
                errors[v["id"]] = str(e)
            job.update(progress=i)
        ids = [v["id"] for v in order if v["id"] in seq]
        # anchor prefilter: all frames of every video in one array
        all_h = np.concatenate([seq[i][0] for i in ids]) if ids else np.zeros(0, np.uint64)
        all_v = np.concatenate([seq[i][1] for i in ids]) if ids else np.zeros(0, bool)
        owner = np.concatenate([np.full(len(seq[i][0]), k) for k, i in enumerate(ids)]) if ids else np.zeros(0, int)
        job.update(message="looking for trimmed copies", progress=0, total=len(ids))
        for k, sid in enumerate(ids):
            job.check()
            job.update(progress=k + 1)
            sh, sv = seq[sid]
            if _dur(by_id[sid]) < p["min_contain_s"] or sv.sum() < 3:
                continue
            anchors = np.flatnonzero(sv)[np.linspace(0, sv.sum() - 1, 3).astype(int)]
            cand = None
            for a_idx in anchors:
                hit = (hamming(sh[a_idx], all_h) <= p["contain_threshold"]) & all_v
                owners = set(owner[hit].tolist()) - {k}
                cand = owners if cand is None else cand & owners
            for lk in cand or ():
                lid = ids[lk]
                if _dur(by_id[lid]) - _dur(by_id[sid]) <= _tol(_dur(by_id[sid]), _dur(by_id[lid]), p):
                    continue  # same length: tier 2's business
                if uf.find(sid) == uf.find(lid):
                    continue
                d, off = align(sh, sv, *seq[lid])
                if d <= p["contain_threshold"]:
                    contained.append({"short": sid, "long": lid, "offset": off, "score": round(d, 2)})

    # assemble sets
    comps = defaultdict(list)
    for vid in list(uf.p):
        comps[uf.find(vid)].append(vid)
    root_pref = {config.norm(r["path"]): i for i, r in enumerate(config.roots())}
    sets = []
    for n, members in enumerate(sorted((m for m in comps.values() if len(m) > 1),
                                       key=lambda m: -max(by_id[x]["size"] for x in m)), 1):
        ms = sorted((by_id[x] for x in members), key=lambda v: keeper_rank(v, root_pref))
        kind = "exact" if len({shas.get(x) for x in members}) == 1 and members[0] in shas else "near"
        reasons = [{"a": a, "b": b, "why": why} for pair, why in evidence.items()
                   for a, b in [tuple(pair)] if a in members and b in members]
        members_out = []
        for v in ms:
            b = _brief(v)
            b["identical"] = v is not ms[0] and shas.get(v["id"]) is not None                 and shas.get(v["id"]) == shas.get(ms[0]["id"])
            members_out.append(b)
        sets.append({"id": f"s{n}", "kind": kind, "keeper": ms[0]["id"],
                     "members": members_out, "reasons": reasons,
                     "wasted": sum(v["size"] for v in ms[1:])})
    # a trim found inside several copies of the same video is one finding:
    # keep one entry per (short, duplicate set of long), preferring the keeper
    keepers = {s["keeper"] for s in sets}
    best: dict[tuple, dict] = {}
    for c in contained:
        k = (uf.find(c["short"]), uf.find(c["long"]))
        cur = best.get(k)
        if cur is None or (c["long"] in keepers, -c["score"]) > (cur["long"] in keepers, -cur["score"]):
            best[k] = c
    contained = list(best.values())
    report = {
        "id": time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2), "created": time.time(), "params": p,
        "deep": deep, "audio": audio, "audio_note": audio_note,
        "scanned": len(videos), "sets": sets,
        "contained": [{**c, "short_v": _brief(by_id[c["short"]]), "long_v": _brief(by_id[c["long"]])}
                      for c in contained],
        "errors": {k: {"error": e, "path": by_id[k]["path"] if k in by_id else k} for k, e in errors.items()},
        "wasted": sum(s["wasted"] for s in sets),
    }
    _save(report)
    job.update(message=f"{len(sets)} duplicate sets", progress=job.total, total=job.total)
    return {"report": report["id"], "sets": len(sets), "contained": len(contained)}


def _brief(v: dict) -> dict:
    i = v["info"]
    return {"id": v["id"], "name": v["name"], "path": v["path"], "rel": v["rel"], "root": v["root"],
            "size": v["size"], "mtime": v["mtime"], "duration": round(_dur(v), 2),
            "width": i["width"], "height": i["height"], "bitrate": int(_bitrate(v)),
            "vcodec": i.get("vcodec"), "has_audio": i.get("has_audio")}


def _save(report: dict) -> None:
    """Write the report under its id, and as latest unless a newer scan exists."""
    os.makedirs(REPORT_DIR(), exist_ok=True)
    latest = load_report("latest")
    names = [f"{report['id']}.json"]
    if latest is None or latest["id"] <= report["id"]:
        names.append("latest.json")
    for name in names:
        with open(os.path.join(REPORT_DIR(), name), "w", encoding="utf-8") as f:
            json.dump(report, f)


def load_report(report_id: str = "latest") -> dict | None:
    if not report_id.replace("-", "").isalnum():
        raise ValueError("bad report id")
    p = os.path.join(REPORT_DIR(), f"{report_id}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        report = json.load(f)
    # files restored from quarantine since the scan are back in play
    for s in report["sets"]:
        for m in s["members"]:
            if m.get("removed"):
                v = library.get(m["id"])
                if v and v["present"] and os.path.exists(v["path"]):
                    m.pop("removed")
    return report


# --- removal -------------------------------------------------------------------------

def remove(selections: list[dict], report_id: str = "latest") -> dict:
    """Quarantine the chosen extras. selections: [{set, keeper, remove: [ids]}].

    Validated against the saved report: every removed id must belong to the
    set, the keeper must too, must not be removed, and must still exist.
    """
    report = load_report(report_id)
    if not report:
        raise LookupError("no duplicate report; run a scan first")
    sets = {s["id"]: s for s in report["sets"]}
    to_move, reasons = [], {}
    for sel in selections:
        s = sets.get(sel.get("set"))
        if not s:
            raise ValueError(f"unknown set {sel.get('set')!r}")
        members = {m["id"]: m for m in s["members"]}
        keeper, rm = sel.get("keeper"), list(dict.fromkeys(sel.get("remove") or []))
        if keeper not in members:
            raise ValueError(f"keeper is not in set {s['id']}")
        if keeper in rm:
            raise ValueError(f"set {s['id']}: the keeper can't also be removed")
        if not os.path.exists(members[keeper]["path"]):
            raise ValueError(f"set {s['id']}: keeper file is missing, refusing to remove its copies")
        for vid in rm:
            if vid not in members:
                raise ValueError(f"{vid} is not in set {s['id']}")
            v = library.get(vid)
            if not v or not v["present"] or not os.path.exists(v["path"]):
                continue  # already gone
            to_move.append(v)
            reasons[vid] = f"duplicate of {members[keeper]['path']}"
    if not to_move:
        return {"batch": None, "moved": 0, "failed": {}}
    res = fileops.quarantine(to_move, reasons)
    moved = {v["id"] for v in to_move} - set(res["failed"])
    for s in report["sets"]:
        for m in s["members"]:
            if m["id"] in moved:
                m["removed"] = res["batch"]
    _save(report)
    return res


def auto_selections(report: dict) -> list[dict]:
    """Every non-keeper of every set (used by the CLI's --apply)."""
    return [{"set": s["id"], "keeper": s["keeper"],
             "remove": [m["id"] for m in s["members"] if m["id"] != s["keeper"] and not m.get("removed")]}
            for s in report["sets"]]
