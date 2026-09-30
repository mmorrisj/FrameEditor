"""FrameKit command line. Run `python -m framekit --help`.

Folder arguments (dupes, group) are indexed on the fly, so the CLI works on
any directory without configuring it as a library root first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import config, fileops, jobs, library, media


def _p(*a, **k):
    print(*a, **k, flush=True)


def _dirs(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        if not os.path.isdir(p):
            sys.exit(f"not a directory: {p}")
        out.append(os.path.abspath(p))
    return out


def _index(dirs: list[str] | None) -> list[str] | None:
    """Scan the given folders (or all configured roots); return the scope."""
    roots = [{"name": os.path.basename(d.rstrip("\\/")) or d, "path": d} for d in dirs] if dirs else None
    r = jobs.run_sync(library.scan, roots=roots, label="indexing")
    _p(f"indexed {r['files']} videos ({r['probed']} new or changed, {r['errors']} unreadable)")
    return dirs


def _video_files(targets: list[str]) -> list[str]:
    files = []
    for t in targets:
        if os.path.isdir(t):
            for dp, _, fs in os.walk(t):
                files += [os.path.join(dp, f) for f in sorted(fs)
                          if os.path.splitext(f)[1].lower() in config.VIDEO_EXTS]
        elif os.path.isfile(t):
            files.append(t)
        else:
            sys.exit(f"no such file or folder: {t}")
    return files


def _library_video(path: str) -> dict:
    """Index a single file (under its own folder) and return its library entry."""
    path = os.path.abspath(path)
    v = library.add_path(path, root=os.path.dirname(path))
    if not v["info"]:
        sys.exit(f"cannot read {path}: {v['error']}")
    return v


# --- commands ------------------------------------------------------------------------

def cmd_roots(a):
    if a.action == "add":
        config.add_root(a.path)
    elif a.action == "remove":
        config.remove_root(a.path)
    for r in config.roots():
        _p(f"{r['name']:<16} {r['path']}{'' if os.path.isdir(r['path']) else '   (missing)'}")


def cmd_scan(a):
    _index(None)


def cmd_list(a):
    for v in library.list_videos(include_errors=True):
        i = v["info"]
        desc = f"{i['width']}x{i['height']} {i['duration']:.1f}s" if i else f"ERROR {v['error']}"
        _p(f"{v['id']}  {desc:<22} {v['path']}")


def cmd_frames(a):
    from .tools import frames
    for src in _video_files(a.targets):
        stem = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(a.out or os.path.dirname(os.path.abspath(src)), f"{stem}-frames-{a.mode}")
        if os.path.exists(os.path.join(out, "manifest.json")) and not a.force:
            _p(f"skip {src}: {out} already exists (use --force)")
            continue
        r = jobs.run_sync(frames.extract, src, out, mode=a.mode, value=a.value, fmt=a.format,
                          width=a.width, limit=a.limit, thumbs=not a.no_thumbs, label=os.path.basename(src))
        _p(f"{r['count']:>6} frames  {out}")


def cmd_audio(a):
    from .tools import audio
    failed = 0
    for src in _video_files(a.targets):
        out = a.out or os.path.dirname(os.path.abspath(src))
        try:
            p = jobs.run_sync(audio.extract, src, out, fmt=a.format, stream=a.track - 1,
                              label=os.path.basename(src))
            _p(f"ok    {p}")
        except ValueError as e:  # e.g. no audio track: expected when sweeping a folder
            _p(f"skip  {src}: {e}")
        except RuntimeError as e:
            failed += 1
            _p(f"FAIL  {src}: {e}")
    if failed:
        sys.exit(1)


def cmd_dupes(a):
    from .tools import dupes
    dirs = _index(_dirs(a.dirs) if a.dirs else None)
    r = jobs.run_sync(dupes.scan, under=dirs, deep=a.deep, audio=a.audio,
                      near_threshold=a.threshold, label="duplicates")
    rep = dupes.load_report(r["report"])
    if a.json:
        _p(json.dumps(rep, indent=1))
        return
    if rep.get("audio_note"):
        _p("note:", rep["audio_note"])
    for s in rep["sets"]:
        _p(f"\n[{s['id']}] {'identical files' if s['kind'] == 'exact' else 'same video, different encode'}")
        for m in s["members"]:
            tag = "KEEP  " if m["id"] == s["keeper"] else ("extra=" if m.get("identical") else "extra ")
            _p(f"  {tag}{m['width']}x{m['height']} {m['bitrate'] // 1000:>6} kb/s {m['size'] / 1e6:>9.1f} MB  {m['path']}")
    for c in rep["contained"]:
        _p(f"\n[trim] {c['short_v']['path']}\n       appears at {c['offset']}s in {c['long_v']['path']}")
    _p(f"\n{len(rep['sets'])} duplicate sets, {rep['wasted'] / 1e6:.1f} MB in extras"
       + (f", {len(rep['contained'])} trimmed copies (never auto-removed)" if rep["contained"] else ""))
    for e in rep["errors"].values():
        _p(f"unreadable: {e['path']}: {e['error']}")
    if not rep["sets"]:
        return
    if a.apply:
        res = dupes.remove(dupes.auto_selections(rep), rep["id"])
        _p(f"moved {res['moved']} extras to quarantine batch {res['batch']}  (undo: python -m framekit quarantine undo {res['batch']})")
        for vid, err in res["failed"].items():
            _p(f"failed: {vid}: {err}")
    else:
        _p("dry run: nothing moved. Re-run with --apply to quarantine every 'extra'.")


def cmd_quarantine(a):
    if a.action == "list":
        bs = fileops.batches()
        if not bs:
            _p("quarantine is empty")
        for b in bs:
            live = b["files"] - (b["restored"] or 0) - (b["purged"] or 0)
            _p(f"{b['batch']}  {b['label']:<10} {b['files']:>4} files, {live} held")
            if a.verbose:
                for f in fileops.batch_files(b["batch"]):
                    _p(f"    {'restored' if f['restored'] else 'deleted' if f['purged'] else 'held':<8} {f['src']}")
    elif a.action == "undo":
        r = fileops.undo(a.batch)
        _p(f"restored {r['restored']} files")
        for s in r["skipped"]:
            _p(f"skipped {s['file']}: {s['why']}")
    elif a.action == "purge":
        if not a.yes:
            sys.exit("purge permanently deletes the batch's files; add --yes to confirm")
        _p(f"deleted {fileops.purge(a.batch)['deleted']} files")


def cmd_group(a):
    from .tools import grouping
    dirs = _index(_dirs(a.dirs) if a.dirs else None)
    r = jobs.run_sync(grouping.build, under=dirs, backend=a.backend, threshold=a.threshold, label="grouping")
    run = grouping.load(r["run"])
    for g in run["groups"]:
        if not g["members"]:
            continue
        _p(f"\n{g['name']} ({len(g['members'])})")
        for vid in g["members"]:
            v = library.get(vid)
            _p(f"   {v['path'] if v else vid}")
    _p(f"\ngrouping {run['id']}: {r['groups']} groups at cutoff {run['threshold']}")
    if a.export:
        content, _ = grouping.export(run["id"], "csv" if a.export.lower().endswith(".csv") else "json")
        with open(a.export, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        _p(f"wrote {a.export}")
    if a.organize:
        res = jobs.run_sync(grouping.organize, run["id"], mode=a.organize, label="organizing")
        _p(f"{a.organize}: {res['done']} files" + (f" into {res['folder']}" if res["folder"] else
                                                  f" (undo: python -m framekit quarantine undo {res['batch']})"))


def cmd_analyze(a):
    from .tools import analysis, frames
    v = _library_video(a.video)
    m = jobs.run_sync(frames.extract_video, v["id"], mode=a.mode, value=a.value, width=a.width,
                      label="extracting")
    _p(f"extracted {m['count']} frames (run {m['id']})")
    r = jobs.run_sync(analysis.analyze, v["id"], m["id"], backend=a.backend,
                      cluster_threshold=a.cluster_threshold, label="analyzing")
    _p(f"{r['shots']} shots, {r['clusters']} clusters, {r['dupes']} duplicate frames")
    if a.order != "chronological":
        analysis.set_order(v["id"], m["id"], a.order)
    for kind in a.export or []:
        if kind == "sheet":
            p = analysis.contact_sheet(v["id"], m["id"])
        elif kind == "video":
            p = jobs.run_sync(analysis.render, v["id"], m["id"], fps=a.fps, label="rendering")
        else:
            p = analysis.export_zip(v["id"], m["id"], kind)
        _p(f"wrote {p}")
    _p(f"browse it in the web app at /video/{v['id']}/run/{m['id']}")


def cmd_unique(a):
    from .tools import unique
    kw = {"backend": a.backend, "threshold": a.threshold, "bits": a.bits, "prefer": a.prefer}
    if a.watch:
        w = unique.Watcher()
        _p(f"watching {unique.inbox()} (Ctrl+C to stop)")

        def run(fn, path, label):
            res = jobs.run_sync(fn, path, label=label)
            _p(f"{label}: {res['total']} images, kept {res['kept']}, moved {res.get('moved', 0)} to quarantine"
               + (f", {len(res['skipped'])} unreadable" if res["skipped"] else ""))
        import time
        while True:
            w.poll(run=run)
            time.sleep(w.interval)
    folders = _dirs(a.dirs) if a.dirs else [s["path"] for s in unique.list_sets() if s["count"]]
    for folder in folders:
        res = jobs.run_sync(unique.process, folder, dry_run=not a.apply, label=os.path.basename(folder), **kw)
        _p(f"\n{folder}: {res['total']} images, keeps {res['kept']}, "
           + (f"moved {res['moved']} to quarantine batch {res['batch']}" if a.apply else f"would remove {res['removed']}"))
        if not a.apply:
            for pr in res["pairs"][: None if a.verbose else 10]:
                _p(f"   {os.path.basename(pr['path'])}  looks like  {os.path.basename(pr['kept'])}  ({pr['distance']})")
            if res["removed"] > 10 and not a.verbose:
                _p(f"   ... {res['removed'] - 10} more (-v to list all)")
        for p, err in res["skipped"].items():
            _p(f"   unreadable: {p}: {err}")
    if not a.apply and folders:
        _p("\ndry run: nothing moved. Re-run with --apply to move near-duplicates to quarantine.")
    if not folders:
        _p(f"nothing to do: {unique.inbox()} has no images")


def cmd_reverse(a):
    from .tools import reverse
    for src in _video_files(a.targets):
        name = reverse.output_name(src, a.audio, a.speed, a.boomerang)
        out = os.path.join(a.out or os.path.dirname(os.path.abspath(src)), name)
        if os.path.abspath(out) == os.path.abspath(src):
            sys.exit("refusing to overwrite the source video")
        r = jobs.run_sync(reverse.reverse, src, out, audio=a.audio, speed=a.speed,
                          boomerang=a.boomerang, label=os.path.basename(src))
        _p(f"ok  {out}  ({r['duration']:.1f}s)")


def cmd_resize(a):
    from .tools import resize
    if a.size:
        try:
            a.width, a.height = (int(x) for x in a.size.lower().split("x"))
        except ValueError:
            sys.exit("--size must look like 1080x1920")
    opts = {"preset": a.preset, "width": a.width, "height": a.height, "percent": a.percent,
            "fit": a.fit, "anchor": a.anchor, "quality": a.quality}
    for src in _video_files(a.targets):
        info = media.probe(src)
        out = os.path.join(a.out or os.path.dirname(os.path.abspath(src)),
                           resize.output_name(src, info, **{k: v for k, v in opts.items() if k != "quality"}))
        if os.path.abspath(out) == os.path.abspath(src):
            sys.exit("refusing to overwrite the source video")
        r = jobs.run_sync(resize.resize, src, out, info=info, label=os.path.basename(src), **opts)
        _p(f"ok  {r['from'][0]}x{r['from'][1]} -> {r['to'][0]}x{r['to'][1]}  {out}")


def cmd_scenes(a):
    from .tools import scenes
    for src in _video_files(a.targets):
        v = _library_video(src)
        jobs.run_sync(scenes.detect, v["id"], label=f"scanning {v['name']}")
        state = dict(scenes.load_state(v["id"]))
        if a.threshold is not None:
            state["threshold"] = a.threshold
        if a.min_length is not None:
            state["min_len"] = a.min_length
        c = scenes.cuts(v["id"], state)
        _p(f"\n{src}: {len(c['scenes'])} scenes (sensitivity {state['threshold']}, min {state['min_len']}s)")
        for s in c["scenes"]:
            _p(f"  {s['n']:3d}  {s['start']:9.2f}s to {s['end']:9.2f}s  ({s['length']:.2f}s)")
        if a.list:
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(a.out or os.path.dirname(os.path.abspath(src)), f"{stem}-scenes")
        r = jobs.run_sync(scenes.split, src, out, c["scenes"], mode="fast" if a.fast else "exact",
                          info=v["info"], label="cutting")
        _p(f"wrote {r['clips']} clips to {out}")


def _natural(p: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(p))]


def cmd_colormatch(a):
    import shutil
    from .tools import colormatch as cm
    segs = []
    for t in a.segments:  # a folder contributes its videos in natural order (seg2 before seg10)
        segs += sorted(_video_files([t]), key=_natural) if os.path.isdir(t) else _video_files([t])
    if not segs:
        sys.exit("no segment videos found")
    ends = None
    if a.ends:
        ends = [None if v.strip().lower() in ("", "auto", "last", "full") else int(v) for v in a.ends.split(",")]
    overlaps = None
    if a.overlap:
        vals = [None if v.strip() in ("", "auto") else int(v) for v in a.overlap.split(",")]
        overlaps = [None] + vals
    _p(f"{len(segs)} segments:")
    for k, s in enumerate(segs, 1):
        _p(f"  {k:3d}  {s}")
    r = jobs.run_sync(cm.analyze, segs, reference=a.reference, method=a.method, fit_frames=a.fit_frames,
                      mode=a.mode, strength=a.strength, luma_strength=a.brightness_strength,
                      color_strength=a.color_strength, fps=a.fps, ends=ends, overlaps=overlaps, label="analysing")
    s = cm.load(r["session"])
    for w in s["warnings"]:
        _p(f"warning: {w}")
    _p(f"output {s['stats']['fps']:g} fps; fits: " + ", ".join(f"{k + 1}: {sg['fit']}" for k, sg in enumerate(s["segments"])))
    for j in s["joins"]:
        how = "set by hand" if j["forced"] else ("detected" if j["detected"] else "none found")
        _p(f"  join {j['join']}->{j['join'] + 1}: {j['used']} repeated frame(s) dropped ({how})")
    st, b = s["stats"], s["stats"]["bounds"] + [len(s["stats"]["before_luma"])]
    _p(f"per segment: brightness, and average R G B (reference {st['ref_luma']:.1f}, {st['rgb_ref']}):")
    for k in range(len(segs)):
        bl, al = st["before_luma"][b[k]:b[k + 1]], st["after_luma"][b[k]:b[k + 1]]
        if bl:
            rb, ra = st["rgb_before"][k], st["rgb_after"][k]
            _p(f"  {k + 1:3d}  {sum(bl) / len(bl):6.1f} -> {sum(al) / len(al):6.1f}   "
               f"RGB {rb} -> {ra}  (shift {[round(y - x, 1) for x, y in zip(rb, ra)]})")
    if a.analyze_only:
        _p(f"session {r['session']} (open it in the web app to preview and render)")
        return
    out = jobs.run_sync(cm.render, r["session"], lossless=a.lossless, crossfade=a.crossfade,
                        keep_frames=a.keep_frames, segment_clips=a.clips, label="rendering")
    src_dir = cm._root(r["session"])
    ext = os.path.splitext(out["joined"])[1]
    dest = a.out or os.path.join(os.path.dirname(os.path.abspath(segs[0])), out["joined"])
    if not os.path.splitext(dest)[1]:
        dest += ext
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    shutil.copy2(os.path.join(src_dir, out["joined"]), dest)
    extra = os.path.splitext(dest)[0] + "-handoff"
    shutil.copytree(os.path.join(src_dir, "handoff"), extra, dirs_exist_ok=True)
    for c in cm.load(r["session"])["outputs"]["clips"]:  # segment_02-colormatched.mp4 -> <out>-segment_02.mp4
        shutil.copy2(os.path.join(src_dir, c), f"{os.path.splitext(dest)[0]}-{c.split('-')[0]}{ext}")
    _p(f"wrote {dest} ({out['frames']} frames); corrected last frames in {extra}")
    if a.keep_frames:
        _p(f"PNG frames kept in {os.path.join(src_dir, 'frames')}")


def cmd_colormatch_image(a):
    from .tools import colormatch as cm
    out = a.out or os.path.splitext(a.image)[0] + "-matched.png"
    st = a.strength
    if a.brightness_strength is not None or a.color_strength is not None:
        st = (a.strength if a.brightness_strength is None else a.brightness_strength,
              a.strength if a.color_strength is None else a.color_strength)
    cm.correct_image(a.image, a.reference, out, a.method, st)
    _p(f"wrote {out}")


def cmd_lineage(a):
    from .tools import lineage as ln
    scope = os.path.abspath(a.folder) if getattr(a, "folder", None) else None
    if a.action == "folders":
        for f in ln.folders():
            _p(f)
        return
    if a.action == "add":
        for f in a.args:
            ln.add_folder(f)
        _p("\n".join(ln.folders()))
        return
    if a.action == "remove":
        for f in a.args:
            ln.remove_folder(f)
        return
    if a.action == "scan":
        only = [os.path.abspath(f) for f in a.args] or None
        if only:
            scope = only[0] if len(only) == 1 else scope
        r = jobs.run_sync(ln.scan, only=only, label="indexing")
        _p(f"{r['probed']} clip(s) indexed, {r['moved']} moved, {r['missing']} gone"
           + (f", {r['waiting']} still being written" if r["waiting"] else ""))
        for e in r["errors"]:
            _p(f"  ! {e}")
        f = ln.forest(scope)
        _p(f"{len(f['roots'])} chain(s), {len(f['groups'])} step(s), {len(f['dup_of'])} duplicate(s)")
        return
    if a.action in ("end", "export"):
        if len(a.args) != 2:
            sys.exit(f"usage: lineage {a.action} CLIP FRAME" + (" (or 'full')" if a.action == "end" else ""))
        cid = ln.resolve(a.args[0], scope)
        if cid is None:
            sys.exit(f"nothing matches {a.args[0]!r}; use a label like c007_g03_t2 or a file name")
        if a.action == "export":
            _p(f"saved {ln.export_frame(cid, int(a.args[1]), scope)}")
            return
        end = None if a.args[1].lower() in ("full", "last", "none") else int(a.args[1])
        r = ln.set_end(cid, end)
        _p(f"{ln.forest(scope)['labels'].get(cid)}: " + (f"ends at frame {r['end']} of {r['frames']}" if r["end"]
                                                          else f"uses the whole clip ({r['frames']} frames)"))
        return
    if a.action == "faces":
        number = int(a.args[0].lower().lstrip("c")) if a.args else None
        r = jobs.run_sync(ln.check_faces, number=number, scope=scope, label="finding faces")
        _p(f"{r['checked']} clip(s) checked, {r['with_faces']} with faces; likeness to each chain's "
           f"original face in brackets (55+ on model, 40-55 drifting, below 40 off model):")
        _p("\n".join(ln.tree_lines(scope, number, with_faces=True)))
        return
    if a.action == "tree":
        number = None
        if a.args:
            t = a.args[0].lower().lstrip("c")
            if not t.isdigit():
                sys.exit("give a chain number, e.g. c007 or 7")
            number = int(t)
        _p("\n".join(ln.tree_lines(scope, number, with_faces=a.faces)) or "no chains (index a folder first)")
        return
    if a.action in ("link", "unlink", "reset"):
        need = 1 if a.action == "reset" else 2
        if len(a.args) != need:
            sys.exit(f"usage: lineage {a.action} CHILD" + (" PARENT" if need == 2 else ""))
        ids = []
        for t in a.args:
            cid = ln.resolve(t, scope)
            if cid is None:
                sys.exit(f"nothing matches {t!r}; use a label like c007_g03_t2 or a file name")
            ids.append(cid)
        ln.override(ids[0], ids[1] if need == 2 else None, a.action)
        f = ln.forest(scope)
        g = f["group_of"][f["dup_of"].get(ids[0], ids[0])]
        _p("\n".join(ln.tree_lines(scope, g.chain)))
        return
    sys.exit(f"unknown action {a.action}")


def cmd_dirs(a):
    """Show where everything lives, after .env and environment variables are applied."""
    for k, v in config.directories().items():
        if isinstance(v, list):
            _p(f"{k}: {'; '.join(v) if v else '(none)'}")
        else:
            _p(f"{k}: {v}")
    env = os.environ.get("FRAMEKIT_ENV") or os.path.join(config.REPO, ".env")
    _p(f".env: {env} ({'found' if os.path.isfile(env) else 'not found'})")


def _sound_source(path: str) -> str:
    """A cutter source key for a file on disk: videos go through the library, audio files are copied in."""
    from .tools import sounds
    if not os.path.isfile(path):
        sys.exit(f"no such file: {path}")
    if os.path.splitext(path)[1].lower() in config.VIDEO_EXTS:
        return "v:" + _library_video(path)["id"]
    for x in sounds.list_sources():  # already copied in (same name and size)?
        if x["kind"] == "audio file" and x["name"] == os.path.basename(path) and x.get("size") == os.path.getsize(path):
            return x["key"]
    with open(path, "rb") as f:
        return sounds.add_source(os.path.basename(path), f)


def cmd_sounds(a):
    from .tools import sounds
    if a.action == "list":
        rows = sounds.list_sounds(a.category, " ".join(a.args) or None)
        for x in rows:
            _p(f"{x['id']:5d}  {x['category']:10} {x['duration']:7.2f}s  {x['name']}"
               + (f"  [{', '.join(x['tags'])}]" if x["tags"] else "") + f"  {x['path']}")
        _p(f"{len(rows)} sound(s) in {sounds.sounds_dir()}")
        return
    if a.action == "rescan":
        r = jobs.run_sync(sounds.rescan, label="rescanning")
        _p(f"added {r['added']}, removed {r['removed']}")
        return
    if not a.args:
        sys.exit(f"usage: sounds {a.action} FILE" + (" START END" if a.action == "cut" else ""))
    key = _sound_source(a.args[0])
    jobs.run_sync(sounds.analyze, key, label="analysing")
    opts = {"trim": not a.no_trim, "normalize": a.normalize, "below": a.below}
    tags = [t.strip() for t in (a.tags or "").split(",") if t.strip()]
    if a.action == "cut":
        if len(a.args) != 3:
            sys.exit("usage: sounds cut FILE START END (seconds)")
        x = sounds.save(key, float(a.args[1]), float(a.args[2]), a.name or "", a.category or "other", tags,
                        loop=a.loop, **opts)
        _p(f"saved {x['path']} ({x['duration']:.2f}s)")
        return
    r = sounds.analysis(key, a.below, a.min_gap)  # split
    stem = a.name or os.path.splitext(os.path.basename(a.args[0]))[0]
    _p(f"{len(r['sounds'])} sound(s) found (gap threshold {r['threshold']} dBFS)")
    if a.dry_run:
        for k, x in enumerate(r["sounds"], 1):
            _p(f"  {k:3d}  {x['start']:8.3f}s to {x['end']:8.3f}s  ({x['end'] - x['start']:.2f}s, {x['level']} dB)")
        return
    for k, x in enumerate(r["sounds"], 1):
        y = sounds.save(key, x["start"], x["end"], f"{stem} {k:02d}", a.category or "other", tags, loop=a.loop, **opts)
        _p(f"  saved {y['path']} ({y['duration']:.2f}s)")


def cmd_serve(a):
    if a.log:  # background runs (pythonw) have no console: send all output to a file
        import logging
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        sys.stdout = sys.stderr = open(a.log, "a", buffering=1, encoding="utf-8")
        logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                            format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from .web import create_app
    app = create_app()
    try:
        from waitress import serve  # production server; works on Windows
    except ImportError:
        _p("waitress not installed, using Flask's development server")
        app.run(host=a.host, port=a.port, threaded=True)
        return
    _p(f"FrameKit on http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}")
    serve(app, host=a.host, port=a.port, threads=8, max_request_body_size=8 * 1024**3)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="framekit", description="Video frame, audio, duplicate and grouping tools.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default=os.environ.get("FRAMEKIT_HOST") or "127.0.0.1",
                   help="use 0.0.0.0 to allow other machines on your network (there is no login); "
                        "default $FRAMEKIT_HOST or 127.0.0.1")
    s.add_argument("--port", type=int, default=int(os.environ.get("FRAMEKIT_PORT") or 8082),
                   help="default $FRAMEKIT_PORT or 8082")
    s.add_argument("--log", help="append all output to this file (for running in the background)")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("dirs", help="show every configured directory (after .env and environment variables)")
    s.set_defaults(fn=cmd_dirs)

    s = sub.add_parser("roots", help="list, add or remove library folders")
    s.add_argument("action", nargs="?", choices=["list", "add", "remove"], default="list")
    s.add_argument("path", nargs="?")
    s.set_defaults(fn=cmd_roots)

    s = sub.add_parser("scan", help="index every video in the library folders")
    s.set_defaults(fn=cmd_scan)
    s = sub.add_parser("list", help="list indexed videos with their ids")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("frames", help="extract frames from video files or folders")
    s.add_argument("targets", nargs="+")
    s.add_argument("--mode", choices=["all", "nth", "fps", "scene", "keyframes"], default="scene")
    s.add_argument("--value", type=float, help="N for nth, rate for fps, threshold for scene")
    s.add_argument("--format", choices=["jpg", "png", "webp"], default="jpg")
    s.add_argument("--width", type=int, help="scale frames down to this width")
    s.add_argument("--limit", type=int, help="stop after this many frames")
    s.add_argument("--out", help="parent folder for output (default: next to each video)")
    s.add_argument("--no-thumbs", action="store_true")
    s.add_argument("--force", action="store_true", help="re-extract even if output exists")
    s.set_defaults(fn=cmd_frames)

    s = sub.add_parser("audio", help="extract audio from video files or folders")
    s.add_argument("targets", nargs="+")
    s.add_argument("--format", choices=["copy", "wav", "flac", "mp3"], default="copy")
    s.add_argument("--track", type=int, default=1, help="audio track number (default 1)")
    s.add_argument("--out", help="output folder (default: next to each video)")
    s.set_defaults(fn=cmd_audio)

    s = sub.add_parser("reverse", help="make rewound (reversed) copies of videos")
    s.add_argument("targets", nargs="+", help="video files or folders")
    s.add_argument("--audio", choices=["reverse", "keep", "none"], default="reverse",
                   help="reverse the audio too (default), keep it playing forwards, or drop it")
    s.add_argument("--speed", type=float, default=1.0, help="e.g. 2 for a fast rewind")
    s.add_argument("--boomerang", action="store_true", help="play forwards, then rewind")
    s.add_argument("--out", help="output folder (default: next to each video)")
    s.set_defaults(fn=cmd_reverse)

    s = sub.add_parser("resize", help="resize videos without stretching the picture")
    s.add_argument("targets", nargs="+", help="video files or folders")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", choices=["50pct", "2160p", "1080p", "720p", "480p", "vertical", "square",
                                        "portrait45", "wide1080"])
    g.add_argument("--height", type=int, help="new height; width follows the video's shape")
    g.add_argument("--width", type=int, help="new width; height follows the video's shape")
    g.add_argument("--percent", type=float, help="scale both sides, e.g. 50")
    g.add_argument("--size", help="exact frame size such as 1080x1920 (see --fit)")
    s.add_argument("--fit", choices=["pad", "blur", "crop", "inside"], default="pad",
                   help="for an exact size of a different shape: bars, blurred background, crop, or fit inside")
    s.add_argument("--anchor", choices=["center", "top", "bottom", "left", "right"], default="center",
                   help="which part to keep with --fit crop")
    s.add_argument("--quality", choices=["high", "balanced", "small"], default="high")
    s.add_argument("--out", help="output folder (default: next to each video)")
    s.set_defaults(fn=cmd_resize)

    s = sub.add_parser("scenes", help="split videos into one clip per scene")
    s.add_argument("targets", nargs="+", help="video files or folders")
    s.add_argument("--threshold", type=float, help="scene-change sensitivity 0-1; lower = more cuts (default 0.3)")
    s.add_argument("--min-length", type=float, help="ignore scenes shorter than this many seconds (default 1)")
    s.add_argument("--fast", action="store_true", help="no re-encode; cuts move to the next keyframe")
    s.add_argument("--list", action="store_true", help="only print the scenes, don't write clips")
    s.add_argument("--out", help="parent folder for the clips (default: next to each video)")
    s.set_defaults(fn=cmd_scenes)

    from .tools.colormatch import METHODS as CM_METHODS
    s = sub.add_parser("colormatch", help="fix color drift across chained AI video segments and join them")
    s.add_argument("segments", nargs="+", help="segment videos in order, or a folder of them (sorted by name)")
    s.add_argument("--reference", help="the original start image (default: first frame of segment 1)")
    s.add_argument("--method", choices=list(CM_METHODS), default="exact",
                   help="exact (default): fit from the repeated handoff frame, pixel for pixel; "
                        "falls back to hm-mvgd-hm where there's no matching frame")
    s.add_argument("--fps", default="auto",
                   help="output frame rate: auto (what most segments use, default), first, or a number")
    s.add_argument("--brightness-strength", type=float, help="0-1, how much of the brightness correction to apply")
    s.add_argument("--color-strength", type=float, help="0-1, how much of the colour correction to apply")
    s.add_argument("--mode", choices=["seam", "original"], default="seam",
                   help="seam: match each segment to the end of the previous one (default); "
                        "original: match every segment to the reference")
    s.add_argument("--fit-frames", type=int, default=5, help="frames used to fit each transform (default 5)")
    s.add_argument("--strength", type=float, default=1.0, help="0-1, how much of the correction to apply")
    s.add_argument("--ends", help="frame each segment ends at, comma separated, 'last' for the whole clip "
                                  "(e.g. last,58,last)")
    s.add_argument("--overlap", help="repeated frames per join, comma separated, 'auto' to detect (e.g. 1,1,auto)")
    s.add_argument("--crossfade", type=int, default=0, help="blend this many frames at each join (0-12)")
    s.add_argument("--lossless", action="store_true", help="write a lossless FFV1 .mkv instead of H.264 .mp4")
    s.add_argument("--clips", action="store_true", help="also write each corrected segment as its own clip")
    s.add_argument("--keep-frames", action="store_true", help="keep the corrected PNG frames")
    s.add_argument("--analyze-only", action="store_true", help="print the analysis, don't render")
    s.add_argument("--out", help="output video path (default: next to the first segment)")
    s.set_defaults(fn=cmd_colormatch)

    s = sub.add_parser("colormatch-image", help="color match one image to a reference image")
    s.add_argument("image")
    s.add_argument("reference")
    s.add_argument("--method", choices=list(CM_METHODS), default="exact")
    s.add_argument("--strength", type=float, default=1.0)
    s.add_argument("--brightness-strength", type=float)
    s.add_argument("--color-strength", type=float)
    s.add_argument("--out", help="output PNG (default: <image>-matched.png)")
    s.set_defaults(fn=cmd_colormatch_image)

    s = sub.add_parser("lineage", help="rebuild parent/child chains of AI clips from first and last frames",
                       description="actions: folders | add FOLDER... | remove FOLDER... | scan [FOLDER...] | "
                                   "tree [CHAIN] | faces [CHAIN] | link CHILD PARENT | unlink CHILD PARENT | reset CHILD | "
                                   "end CLIP FRAME|full | export CLIP FRAME")
    s.add_argument("action", choices=["folders", "add", "remove", "scan", "tree", "faces", "link", "unlink", "reset",
                                      "end", "export"])
    s.add_argument("args", nargs="*", help="folders, a chain (c007), or clip labels / file names")
    s.add_argument("--folder", help="work within just this folder instead of every saved one")
    s.add_argument("--faces", action="store_true", help="tree: show each take's likeness to the original face")
    s.set_defaults(fn=cmd_lineage)

    s = sub.add_parser("sounds", help="cut audio into clips for the sound library",
                       description="actions: list [SEARCH] | rescan | split FILE | cut FILE START END")
    s.add_argument("action", choices=["list", "rescan", "split", "cut"])
    s.add_argument("args", nargs="*", help="search words, or a video/audio file (and START END seconds for cut)")
    s.add_argument("--category", help="category to save into (default: other) or to list")
    s.add_argument("--name", help="clip name (split: prefix for the numbered clips)")
    s.add_argument("--tags", help="comma-separated tags")
    s.add_argument("--loop", action="store_true", help="mark the clips as loopable")
    s.add_argument("--below", type=float, default=35.0, help="gaps are this many dB below the loud parts (default 35)")
    s.add_argument("--min-gap", type=float, default=0.3, help="shortest gap that splits sounds, seconds (default 0.3)")
    s.add_argument("--normalize", action="store_true", help="peak-normalise each clip to -1 dBFS")
    s.add_argument("--no-trim", action="store_true", help="keep the exact range instead of trimming silence")
    s.add_argument("--dry-run", action="store_true", help="split: only list what would be saved")
    s.set_defaults(fn=cmd_sounds)

    s = sub.add_parser("dupes", help="find duplicate videos (dry run unless --apply)")
    s.add_argument("dirs", nargs="*", help="folders to check (default: library folders)")
    s.add_argument("--deep", action="store_true", help="also find trimmed copies (full decode)")
    s.add_argument("--audio", action="store_true", help="also compare audio fingerprints (needs fpcalc)")
    s.add_argument("--threshold", type=float, help="near-duplicate cutoff in bits (default 10)")
    s.add_argument("--apply", action="store_true", help="move every extra into quarantine")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_dupes)

    s = sub.add_parser("quarantine", help="list, undo or permanently delete quarantine batches")
    s.add_argument("action", choices=["list", "undo", "purge"], nargs="?", default="list")
    s.add_argument("batch", nargs="?")
    s.add_argument("--yes", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_quarantine)

    s = sub.add_parser("group", help="group videos by similarity of sampled frames")
    s.add_argument("dirs", nargs="*", help="folders to group (default: library folders)")
    s.add_argument("--backend", choices=["visual", "clip"], default="visual")
    s.add_argument("--threshold", type=float, help="distance cutoff (default: visual 0.20, clip 0.10)")
    s.add_argument("--export", help="write the grouping to FILE.json or FILE.csv")
    s.add_argument("--organize", choices=["copy", "move"], help="also put files into a folder per group")
    s.set_defaults(fn=cmd_group)

    s = sub.add_parser("unique", help="remove near-duplicate images from folders (dry run unless --apply)")
    s.add_argument("dirs", nargs="*", help="image folders (default: every set in the frames inbox)")
    s.add_argument("--apply", action="store_true", help="move near-duplicates to quarantine")
    s.add_argument("--watch", action="store_true", help="keep watching the inbox and apply automatically")
    s.add_argument("--backend", choices=["visual", "clip"])
    s.add_argument("--threshold", type=float, help="similarity cutoff (default: visual 0.10, clip 0.015)")
    s.add_argument("--bits", type=int, help="also remove images within this many hash bits (default 4)")
    s.add_argument("--prefer", choices=["sharpest", "largest", "name"])
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_unique)

    s = sub.add_parser("analyze", help="extract frames from a video and cluster/order them")
    s.add_argument("video")
    s.add_argument("--mode", choices=["all", "nth", "fps", "scene", "keyframes"], default="fps")
    s.add_argument("--value", type=float)
    s.add_argument("--width", type=int)
    s.add_argument("--backend", choices=["visual", "clip"], default="visual")
    s.add_argument("--cluster-threshold", type=float)
    s.add_argument("--order", choices=["chronological", "cluster", "chain"], default="chronological")
    s.add_argument("--export", nargs="*", choices=["ordered", "clusters", "sheet", "video"])
    s.add_argument("--fps", type=float, default=24.0, help="frame rate for --export video")
    s.set_defaults(fn=cmd_analyze)

    a = ap.parse_args(argv)
    if a.cmd == "roots" and a.action in ("add", "remove") and not a.path:
        ap.error(f"roots {a.action} needs a PATH")
    if a.cmd == "quarantine" and a.action in ("undo", "purge") and not a.batch:
        ap.error(f"quarantine {a.action} needs a BATCH id (see: quarantine list)")
    config.ensure_dirs()
    try:
        a.fn(a)
    except (ValueError, LookupError, RuntimeError) as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit(130)
