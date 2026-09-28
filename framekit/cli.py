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
    s.add_argument("--host", default="127.0.0.1",
                   help="use 0.0.0.0 to allow other machines on your network (there is no login)")
    s.add_argument("--port", type=int, default=8082)
    s.add_argument("--log", help="append all output to this file (for running in the background)")
    s.set_defaults(fn=cmd_serve)

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
