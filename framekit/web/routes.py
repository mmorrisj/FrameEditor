"""Routes for the library and every suite tool except the editor."""
from __future__ import annotations

import os
import re

from flask import Blueprint, Response, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from .. import config, features, fileops, jobs, library, media, samples
from ..tools import (analysis, audio, colormatch, dupes, frames, grouping, lineage, resize, reverse, scenes,
                     unique)

bp = Blueprint("suite", __name__)

_VID = re.compile(r"^[0-9a-f]{12}$")
_FILE = re.compile(r"^[^/\\]{1,200}$")


def _body() -> dict:
    return request.get_json(silent=True) or {}


def _vid(vid: str) -> dict:
    if not _VID.match(vid):
        abort(400, "bad video id")
    try:
        return library.require(vid)
    except LookupError as e:
        abort(404, str(e))


def _job(j: jobs.Job):
    return jsonify(job=j.id, ref=j.ref)


def _num(d: dict, key: str, cast=float):
    v = d.get(key)
    if v in (None, ""):
        return None
    try:
        return cast(v)
    except (TypeError, ValueError):
        abort(400, f"{key} must be a number")


# --- pages -------------------------------------------------------------------------

@bp.route("/")
def page_library():
    return render_template("library.html", page="library")


@bp.route("/video/<vid>")
def page_video(vid):
    v = _vid(vid)
    return render_template("video.html", page="library", vid=vid, title=v["name"])


@bp.route("/video/<vid>/run/<run>")
def page_run(vid, run):
    v = _vid(vid)
    frames.run_dir(vid, run)
    return render_template("run.html", page="library", vid=vid, run=run, title=v["name"])


@bp.route("/video/<vid>/scenes")
def page_scenes(vid):
    v = _vid(vid)
    return render_template("scenes.html", page="library", vid=vid, title=v["name"])


@bp.route("/dupes")
def page_dupes():
    return render_template("dupes.html", page="dupes")


@bp.route("/groups")
def page_groups():
    return render_template("groups.html", page="groups")


@bp.route("/unique")
def page_unique():
    return render_template("unique.html", page="unique")


@bp.route("/lineage")
def page_lineage():
    return render_template("lineage.html", page="lineage")


@bp.route("/colormatch")
def page_colormatch():
    return render_template("colormatch.html", page="colormatch")


# --- jobs --------------------------------------------------------------------------

@bp.route("/api/jobs")
def api_jobs():
    return jsonify(jobs.list_jobs())


@bp.route("/api/jobs/<jid>/cancel", methods=["POST"])
def api_job_cancel(jid):
    return jsonify(ok=jobs.cancel(jid))


# --- library -------------------------------------------------------------------------

@bp.route("/api/library")
def api_library():
    vids = library.list_videos(include_errors=True)
    for v in vids:
        v["has_samples"] = os.path.exists(samples.sample_path(v["id"], 0))
    return jsonify(roots=config.roots(), videos=vids,
                   backends=features.backends(), audio_fp=dupes.audio_available())


@bp.route("/api/library/scan", methods=["POST"])
def api_scan():
    return _job(jobs.submit("scan", "Scan library folders", library.scan, ref="library"))


@bp.route("/api/library/upload", methods=["POST"])
def api_upload():
    saved = []
    for f in request.files.getlist("video"):
        name = secure_filename(f.filename or "")
        if not name or os.path.splitext(name)[1].lower() not in config.VIDEO_EXTS:
            abort(400, f"unsupported file: {f.filename}")
        dst = fileops._free(os.path.join(config.UPLOADS, name))
        f.save(dst)
        saved.append(library.add_path(dst)["id"])
    return jsonify(ids=saved)


@bp.route("/api/video/<vid>")
def api_video(vid):
    v = _vid(vid)
    try:
        s = samples.get(v) if request.args.get("samples") == "1" else None
    except Exception:
        s = None
    has = os.path.exists(samples.sample_path(vid, 0))
    n = len(s["paths"]) if s else (
        len([f for f in os.listdir(library.work_dir(vid, "samples"))]) if has else 0)
    return jsonify(video=v, samples=n, runs=frames.list_runs(vid), audio=audio.list_outputs(vid),
                   reversed=reverse.list_outputs(vid), reverse_audio=reverse.AUDIO,
                   resized=resize.list_outputs(vid), shown=resize.display_size(v["info"]),
                   resize_presets={k: p[0] for k, p in resize.PRESETS.items()}, resize_fits=resize.FITS,
                   modes=frames.MODES, defaults=frames.DEFAULT_VALUES, audio_formats=audio.FORMATS,
                   active=jobs.active_for(vid))


@bp.route("/api/video/<vid>/samples", methods=["POST"])
def api_samples(vid):
    v = _vid(vid)
    return _job(jobs.submit("samples", f"Sample frames: {v['name']}",
                            lambda job: len(samples.get(v)["paths"]), ref=vid))


@bp.route("/media/sample/<vid>/<int:i>")
def media_sample(vid, i):
    if not _VID.match(vid):
        abort(400)
    p = samples.sample_path(vid, i)
    if not os.path.exists(p):
        abort(404)
    return send_file(p, max_age=3600)


@bp.route("/media/video/<vid>")
def media_video(vid):
    return send_file(_vid(vid)["path"], conditional=True)


# --- frames ----------------------------------------------------------------------------

def _frame_opts(d: dict) -> dict:
    mode = d.get("mode", "scene")
    if mode not in frames.MODES:
        abort(400, "unknown mode")
    fmt = d.get("fmt", "jpg")
    if fmt not in frames.FORMATS:
        abort(400, "unknown format")
    o = {"mode": mode, "value": _num(d, "value"), "fmt": fmt,
         "width": _num(d, "width", int), "limit": _num(d, "limit", int)}
    frames._filters(mode, o["value"], {"fps_frac": None})  # reject bad values now, not mid-job
    return o


@bp.route("/api/video/<vid>/frames", methods=["POST"])
def api_frames(vid):
    v = _vid(vid)
    o = _frame_opts(_body())
    return _job(jobs.submit("frames", f"Frames ({frames.MODES[o['mode']].lower()}): {v['name']}",
                            frames.extract_video, vid, ref=vid, **o))


@bp.route("/api/frames/batch", methods=["POST"])
def api_frames_batch():
    d = _body()
    o = _frame_opts(d)
    out = []
    for vid in d.get("ids", []):
        v = _vid(vid)
        out.append(jobs.submit("frames", f"Frames ({frames.MODES[o['mode']].lower()}): {v['name']}",
                               frames.extract_video, vid, ref=vid, **o).id)
    return jsonify(jobs=out)


@bp.route("/api/video/<vid>/runs/<run>", methods=["GET", "DELETE"])
def api_run(vid, run):
    _vid(vid)
    if request.method == "DELETE":
        frames.delete_run(vid, run)
        return jsonify(ok=True)
    m = frames.get_run(vid, run)
    return jsonify(run=m, analysis=analysis.load(vid, run), backends=features.backends(),
                   defaults=analysis.DEFAULTS, active=jobs.active_for(f"{vid}/{run}"))


@bp.route("/media/run/<vid>/<run>/<kind>/<int:n>")
def media_run(vid, run, kind, n):
    if not _VID.match(vid):
        abort(400)
    d = frames.run_dir(vid, run)
    if kind == "thumb":
        p = os.path.join(d, "thumbs", f"t{n:06d}.jpg")
        if not os.path.exists(p):
            kind = "full"
    if kind == "full":
        hits = [f for f in os.listdir(d) if f.startswith(f"f{n:06d}.")] if os.path.isdir(d) else []
        if not hits:
            abort(404)
        p = os.path.join(d, hits[0])
    elif kind != "thumb":
        abort(404)
    return send_file(p, max_age=3600)


@bp.route("/api/video/<vid>/runs/<run>/zip")
def api_run_zip(vid, run):
    _vid(vid)
    p = frames.zip_run(vid, run)
    return send_file(p, as_attachment=True, download_name=os.path.basename(p))


# --- audio -------------------------------------------------------------------------------

@bp.route("/api/video/<vid>/audio", methods=["POST"])
def api_audio(vid):
    v = _vid(vid)
    d = _body()
    fmt = d.get("fmt", "copy")
    if fmt not in audio.FORMATS:
        abort(400, "unknown audio format")
    return _job(jobs.submit("audio", f"Audio ({fmt}): {v['name']}", audio.extract_video, vid,
                            fmt=fmt, stream=_num(d, "stream", int) or 0, ref=vid))


@bp.route("/api/audio/batch", methods=["POST"])
def api_audio_batch():
    d = _body()
    fmt = d.get("fmt", "copy")
    if fmt not in audio.FORMATS:
        abort(400, "unknown audio format")
    ids = [_vid(x)["id"] for x in d.get("ids", [])]
    return _job(jobs.submit("audio", f"Audio ({fmt}) from {len(ids)} videos", audio.extract_many, ids,
                            fmt=fmt, ref="library"))


@bp.route("/api/video/<vid>/audio/<name>", methods=["GET", "DELETE"])
def api_audio_file(vid, name):
    _vid(vid)
    if not _FILE.match(name) or name.startswith("."):
        abort(400)
    p = os.path.join(audio.out_dir(vid), name)
    if not os.path.isfile(p):
        abort(404)
    if request.method == "DELETE":
        os.remove(p)
        return jsonify(ok=True)
    return send_file(p, as_attachment=True, download_name=name)


# --- duplicates ----------------------------------------------------------------------------

@bp.route("/api/dupes/scan", methods=["POST"])
def api_dupes_scan():
    d = _body()
    kw = {k: _num(d, k) for k in ("near_threshold", "contain_threshold")}
    return _job(jobs.submit("dupes", "Find duplicate videos", dupes.scan,
                            video_ids=d.get("ids") or None, deep=bool(d.get("deep")),
                            audio=bool(d.get("audio")), ref="dupes", **kw))


@bp.route("/api/dupes/report")
def api_dupes_report():
    r = dupes.load_report(request.args.get("id", "latest"))
    return jsonify(report=r, audio_fp=dupes.audio_available(), active=jobs.active_for("dupes"))


@bp.route("/api/dupes/remove", methods=["POST"])
def api_dupes_remove():
    d = _body()
    return jsonify(dupes.remove(d.get("selections") or [], d.get("report", "latest")))


@bp.route("/api/quarantine")
def api_quarantine():
    out = []
    for b in fileops.batches():
        b["files_list"] = fileops.batch_files(b["batch"])
        out.append(b)
    return jsonify(batches=out, folder=config.QUARANTINE)


@bp.route("/api/quarantine/<batch>/undo", methods=["POST"])
def api_undo(batch):
    if not re.match(r"^[0-9-]+[0-9a-f]{4}$", batch):
        abort(400)
    files = fileops.batch_files(batch)
    if files and files[0]["label"] == "unique":
        return jsonify(unique.undo(batch))  # also pins the images so the watcher keeps them
    return jsonify(fileops.undo(batch))


@bp.route("/api/quarantine/<batch>/purge", methods=["POST"])
def api_purge(batch):
    if not re.match(r"^[0-9-]+[0-9a-f]{4}$", batch):
        abort(400)
    if _body().get("confirm") != batch:
        abort(400, "purge must be confirmed with the batch id")
    return jsonify(fileops.purge(batch))


# --- grouping ---------------------------------------------------------------------------

@bp.route("/api/groups", methods=["GET", "POST"])
def api_groups():
    if request.method == "GET":
        return jsonify(runs=grouping.list_runs(), backends=features.backends(),
                       defaults=grouping.DEFAULT_THRESHOLD, active=jobs.active_for("groups"))
    d = _body()
    backend = d.get("backend", "visual")
    if backend not in features.backends():
        abort(400, "unknown backend")
    if not features.backends()[backend]:
        abort(400, f"the {backend} backend is not installed")
    return _job(jobs.submit("groups", f"Group videos ({backend})", grouping.build,
                            video_ids=d.get("ids") or None, backend=backend,
                            threshold=_num(d, "threshold"), ref="groups"))


@bp.route("/api/groups/<rid>", methods=["GET", "DELETE"])
def api_group(rid):
    if request.method == "DELETE":
        grouping.delete(rid)
        return jsonify(ok=True)
    run = grouping.load(rid)
    vids = {}
    for vid in run["videos"]:
        v = library.get(vid)
        if v:
            vids[vid] = {"id": vid, "name": v["name"], "rel": v["rel"], "present": v["present"],
                         "duration": (v["info"] or {}).get("duration"), "size": v["size"]}
    return jsonify(run=run, videos=vids)


@bp.route("/api/groups/<rid>/threshold", methods=["POST"])
def api_group_threshold(rid):
    t = _num(_body(), "threshold")
    if t is None or not 0 < t < 2:
        abort(400, "threshold must be between 0 and 2")
    grouping.recut(rid, t)
    return api_group(rid)


@bp.route("/api/groups/<rid>/move", methods=["POST"])
def api_group_move(rid):
    d = _body()
    grouping.move(rid, d.get("video"), d.get("to"))
    return api_group(rid)


@bp.route("/api/groups/<rid>/rename", methods=["POST"])
def api_group_rename(rid):
    d = _body()
    grouping.rename(rid, d.get("group", ""), d.get("name", ""))
    return api_group(rid)


@bp.route("/api/groups/<rid>/export")
def api_group_export(rid):
    fmt = request.args.get("fmt", "json")
    content, mime = grouping.export(rid, "csv" if fmt == "csv" else "json")
    return Response(content, mimetype=mime, headers={
        "Content-Disposition": f"attachment; filename=grouping-{rid}.{'csv' if fmt == 'csv' else 'json'}"})


@bp.route("/api/groups/<rid>/organize", methods=["POST"])
def api_group_organize(rid):
    d = _body()
    mode = d.get("mode", "copy")
    if mode not in ("copy", "move"):
        abort(400, "mode must be copy or move")
    grouping.load(rid)
    return _job(jobs.submit("organize", f"Organize grouping into folders ({mode})", grouping.organize,
                            rid, mode=mode, include_singles=bool(d.get("singles")), ref="groups"))


# --- frame analysis ---------------------------------------------------------------------------

@bp.route("/api/video/<vid>/runs/<run>/analyze", methods=["POST"])
def api_analyze(vid, run):
    v = _vid(vid)
    d = _body()
    backend = d.get("backend", "visual")
    if not features.backends().get(backend):
        abort(400, f"backend {backend!r} is not available")
    frames.get_run(vid, run)
    return _job(jobs.submit("analysis", f"Analyze frames ({backend}): {v['name']}", analysis.analyze,
                            vid, run, backend=backend, ref=f"{vid}/{run}"))


@bp.route("/api/video/<vid>/runs/<run>/analysis", methods=["POST"])
def api_analysis_update(vid, run):
    _vid(vid)
    d = _body()
    if "params" in d:
        p = d["params"] or {}
        analysis.set_params(vid, run, **{k: _num(p, k) for k in
                                         ("shot_threshold", "cluster_threshold", "dup_threshold")})
    if "order" in d:
        analysis.set_order(vid, run, d["order"], d.get("custom"))
    if "exclude" in d:
        analysis.set_excluded(vid, run, [int(i) for i in d["exclude"]], True)
    if "include" in d:
        analysis.set_excluded(vid, run, [int(i) for i in d["include"]], False)
    return jsonify(analysis=analysis.load(vid, run))


@bp.route("/api/video/<vid>/runs/<run>/export/<kind>")
def api_analysis_export(vid, run, kind):
    _vid(vid)
    if kind in ("ordered", "clusters"):
        p = analysis.export_zip(vid, run, kind)
    elif kind == "sheet":
        p = analysis.contact_sheet(vid, run)
    else:
        abort(404)
    return send_file(p, as_attachment=True, download_name=os.path.basename(p))


@bp.route("/api/video/<vid>/runs/<run>/render", methods=["POST"])
def api_render(vid, run):
    v = _vid(vid)
    fps = _num(_body(), "fps") or 24.0
    if not 0.1 <= fps <= 120:
        abort(400, "fps must be between 0.1 and 120")
    return _job(jobs.submit("render", f"Render ordered frames: {v['name']}", analysis.render,
                            vid, run, fps=fps, ref=f"{vid}/{run}"))


@bp.route("/api/exports/<name>")
def api_export_file(name):
    if not _FILE.match(name) or name.startswith("."):
        abort(400)
    p = os.path.join(config.EXPORTS, name)
    if not os.path.isfile(p):
        abort(404)
    return send_file(p, as_attachment=True, download_name=name)


# --- unique frames (drop folder) ------------------------------------------------------------

def _set(d: dict) -> str:
    return unique.set_path(d.get("set") or "")


@bp.route("/api/unique")
def api_unique():
    w = unique.watcher()
    sets = unique.list_sets()
    for s in sets:
        s["active"] = bool(jobs.active_for(f"unique:{s['path']}"))
    return jsonify(inbox=unique.inbox(), settings=unique.get_settings(), sets=sets,
                   runs=unique.runs(), last=w.last, backends=features.backends(),
                   defaults=unique.DEFAULT_THRESHOLD, prefer=unique.PREFER)


@bp.route("/api/unique/settings", methods=["POST"])
def api_unique_settings():
    d = _body()
    return jsonify(settings=unique.save_settings(
        backend=d.get("backend"), threshold=_num(d, "threshold"), bits=_num(d, "bits", int),
        prefer=d.get("prefer"), watch=d.get("watch")))


@bp.route("/api/unique/upload", methods=["POST"])
def api_unique_upload():
    name = secure_filename(request.form.get("set", "").strip())
    folder = os.path.join(unique.inbox(), name) if name else unique.inbox()
    if name and name.startswith(("_", ".")):
        abort(400, "set names can't start with _ or .")
    os.makedirs(folder, exist_ok=True)
    n = 0
    for f in request.files.getlist("image"):
        fn = secure_filename(f.filename or "")
        if not fn or os.path.splitext(fn)[1].lower() not in unique.IMAGE_EXTS:
            continue
        f.save(fileops._free(os.path.join(folder, fn)))
        n += 1
    if not n:
        abort(400, "no supported images in the upload")
    return jsonify(saved=n, set=name)


@bp.route("/api/unique/run", methods=["POST"])
def api_unique_run():
    d = _body()
    folder = _set(d)
    dry = bool(d.get("preview"))
    if jobs.active_for(f"unique:{folder}"):
        abort(409, "this set is already being processed")
    kw = {"backend": d.get("backend"), "threshold": _num(d, "threshold"),
          "bits": _num(d, "bits", int), "prefer": d.get("prefer")}
    if kw["backend"] and not features.backends().get(kw["backend"]):
        abort(400, f"backend {kw['backend']!r} is not available")
    label = ("Preview" if dry else "Unique frames") + f": {os.path.basename(folder) or 'inbox'}"
    return _job(jobs.submit("unique", label, unique.process, folder, dry_run=dry,
                            ref=f"unique:{folder}", **kw))


@bp.route("/api/unique/runs/<batch>/undo", methods=["POST"])
def api_unique_undo(batch):
    if not re.match(r"^[0-9-]+[0-9a-f]{4}$", batch):
        abort(400)
    return jsonify(unique.undo(batch))


@bp.route("/api/unique/set/zip")
def api_unique_zip():
    folder = unique.set_path(request.args.get("set", ""))
    import zipfile
    os.makedirs(config.EXPORTS, exist_ok=True)
    out = os.path.join(config.EXPORTS, f"unique-{os.path.basename(folder) or 'inbox'}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in unique.list_images(folder):
            z.write(p, os.path.basename(p))
    return send_file(out, as_attachment=True, download_name=os.path.basename(out))


@bp.route("/api/unique/set/images")
def api_unique_images():
    folder = unique.set_path(request.args.get("set", ""))
    return jsonify(images=unique.list_images(folder))


@bp.route("/media/unique")
def media_unique():
    p = request.args.get("p", "")
    if not os.path.isfile(p):
        abort(404)
    try:
        return send_file(unique.thumb(p), max_age=3600)
    except ValueError:
        abort(403)


# --- reverse ----------------------------------------------------------------------------------

def _reverse_opts(d: dict) -> dict:
    a = d.get("audio", "reverse")
    if a not in reverse.AUDIO:
        abort(400, "unknown audio option")
    speed = _num(d, "speed") or 1.0
    if not 0.25 <= speed <= 16:
        abort(400, "speed must be between 0.25 and 16")
    return {"audio": a, "speed": speed, "boomerang": bool(d.get("boomerang")),
            "to_library": bool(d.get("library"))}


@bp.route("/api/video/<vid>/reverse", methods=["POST"])
def api_reverse(vid):
    v = _vid(vid)
    o = _reverse_opts(_body())
    kind = "Boomerang" if o["boomerang"] else "Reverse"
    return _job(jobs.submit("reverse", f"{kind}: {v['name']}", reverse.reverse_video, vid, ref=vid, **o))


@bp.route("/api/reverse/batch", methods=["POST"])
def api_reverse_batch():
    d = _body()
    o = _reverse_opts(d)
    out = []
    for vid in d.get("ids", []):
        v = _vid(vid)
        out.append(jobs.submit("reverse", f"Reverse: {v['name']}", reverse.reverse_video, vid,
                               ref=vid, **o).id)
    return jsonify(jobs=out)


@bp.route("/api/video/<vid>/reversed/<name>", methods=["GET", "DELETE"])
def api_reversed_file(vid, name):
    _vid(vid)
    if not _FILE.match(name) or name.startswith("."):
        abort(400)
    p = os.path.join(reverse.out_dir(vid), name)
    if not os.path.isfile(p):
        abort(404)
    if request.method == "DELETE":
        os.remove(p)
        return jsonify(ok=True)
    return send_file(p, conditional=True, as_attachment=request.args.get("dl") == "1", download_name=name)


# --- resize -------------------------------------------------------------------------------------

def _resize_opts(d: dict) -> dict:
    o = {"preset": d.get("preset") or None, "width": _num(d, "width", int), "height": _num(d, "height", int),
         "percent": _num(d, "percent"), "fit": d.get("fit") or "pad", "anchor": d.get("anchor") or "center",
         "quality": d.get("quality") or "high"}
    if o["preset"]:
        o["width"] = o["height"] = o["percent"] = None
    resize.resolve(o["preset"], o["width"], o["height"], o["percent"])  # validate now, not mid-job
    if o["fit"] not in resize.FITS or o["anchor"] not in resize.ANCHORS or o["quality"] not in resize.QUALITY:
        abort(400, "bad fit, anchor or quality")
    return o


@bp.route("/api/video/<vid>/resize", methods=["POST"])
def api_resize(vid):
    v = _vid(vid)
    d = _body()
    o = _resize_opts(d)
    return _job(jobs.submit("resize", f"Resize: {v['name']}", resize.resize_video, vid,
                            to_library=bool(d.get("library")), ref=vid, **o))


@bp.route("/api/resize/batch", methods=["POST"])
def api_resize_batch():
    d = _body()
    o = _resize_opts(d)
    out = []
    for vid in d.get("ids", []):
        v = _vid(vid)
        out.append(jobs.submit("resize", f"Resize: {v['name']}", resize.resize_video, vid,
                               to_library=bool(d.get("library")), ref=vid, **o).id)
    return jsonify(jobs=out)


@bp.route("/api/video/<vid>/resize/preview", methods=["POST"])
def api_resize_preview(vid):
    v = _vid(vid)
    o = _resize_opts(_body())
    o.pop("quality")
    png = resize.preview(v["path"], media.probe(v["path"]), **o)
    return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store"})


@bp.route("/api/video/<vid>/resized/<name>", methods=["GET", "DELETE"])
def api_resized_file(vid, name):
    _vid(vid)
    if not _FILE.match(name) or name.startswith("."):
        abort(400)
    p = os.path.join(resize.out_dir(vid), name)
    if not os.path.isfile(p):
        abort(404)
    if request.method == "DELETE":
        os.remove(p)
        return jsonify(ok=True)
    return send_file(p, conditional=True, as_attachment=request.args.get("dl") == "1", download_name=name)


# --- scene splitting ---------------------------------------------------------------------------

@bp.route("/api/video/<vid>/scenes")
def api_scenes(vid):
    _vid(vid)
    return jsonify(cuts=scenes.cuts(vid), exports=scenes.list_exports(vid),
                   active=[j for j in jobs.active_for(vid) if j["kind"] == "scenes"])


@bp.route("/api/video/<vid>/scenes/detect", methods=["POST"])
def api_scenes_detect(vid):
    v = _vid(vid)
    return _job(jobs.submit("scenes", f"Find scenes: {v['name']}", lambda job: len(scenes.detect(job, vid)["times"]),
                            ref=vid))


@bp.route("/api/video/<vid>/scenes/settings", methods=["POST"])
def api_scenes_settings(vid):
    _vid(vid)
    d = _body()
    return jsonify(scenes.cuts(vid, scenes.save_state(vid, threshold=_num(d, "threshold"),
                                                      min_len=_num(d, "min_len"))))


@bp.route("/api/video/<vid>/scenes/cut", methods=["POST"])
def api_scenes_cut(vid):
    _vid(vid)
    d = _body()
    action = d.get("action")
    if action == "reset":
        return jsonify(scenes.reset_edits(vid))
    t = _num(d, "t")
    if t is None or t < 0:
        abort(400, "a time is needed")
    if action == "add":
        return jsonify(scenes.add_cut(vid, t))
    if action == "remove":
        return jsonify(scenes.remove_cut(vid, t))
    abort(400, "action must be add, remove or reset")


@bp.route("/api/video/<vid>/scenes/export", methods=["POST"])
def api_scenes_export(vid):
    v = _vid(vid)
    d = _body()
    mode = d.get("mode", "exact")
    if mode not in ("exact", "fast"):
        abort(400, "mode must be exact or fast")
    only = [int(n) for n in d.get("only") or []] or None
    if not scenes.detected(vid):
        abort(400, "find the scenes first")
    return _job(jobs.submit("scenes", f"Split into scenes ({mode}): {v['name']}", scenes.export, vid,
                            mode=mode, only=only, to_library=bool(d.get("library")), ref=vid))


@bp.route("/api/scenes/batch", methods=["POST"])
def api_scenes_batch():
    d = _body()
    mode = d.get("mode", "exact")
    if mode not in ("exact", "fast"):
        abort(400, "mode must be exact or fast")
    out = []
    for vid in d.get("ids", []):
        v = _vid(vid)
        out.append(jobs.submit("scenes", f"Split into scenes: {v['name']}", scenes.split_video, vid,
                               mode=mode, to_library=bool(d.get("library")), ref=vid).id)
    return jsonify(jobs=out)


@bp.route("/api/video/<vid>/scenes/exports/<run>", methods=["DELETE"])
def api_scenes_export_delete(vid, run):
    _vid(vid)
    scenes.delete_export(vid, run)
    return jsonify(ok=True)


@bp.route("/api/video/<vid>/scenes/exports/<run>/zip")
def api_scenes_export_zip(vid, run):
    _vid(vid)
    p = scenes.zip_export(vid, run)
    return send_file(p, as_attachment=True, download_name=os.path.basename(p))


@bp.route("/api/video/<vid>/scenes/exports/<run>/<name>")
def api_scenes_export_file(vid, run, name):
    _vid(vid)
    p = scenes.export_path(vid, run, name)
    if not os.path.isfile(p):
        abort(404)
    return send_file(p, conditional=True)


@bp.route("/media/scene/<vid>/<int:ms>")
def media_scene(vid, ms):
    _vid(vid)
    return send_file(scenes.thumb(vid, ms / 1000), max_age=86400)


# --- color match -------------------------------------------------------------------------------

_REF = re.compile(r"^[0-9a-f]{16}$")
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _cm(fn, *a):
    try:
        return fn(*a)
    except ValueError as e:
        abort(400, str(e))
    except LookupError as e:
        abort(404, str(e))


def _save_upload_image(f, folder: str) -> str:
    """Save an uploaded image as PNG under `folder`; returns its token."""
    if not f or os.path.splitext(f.filename or "")[1].lower() not in _IMG_EXTS:
        abort(400, "upload a JPG, PNG, WebP, BMP or TIFF image")
    from PIL import Image
    token = os.urandom(8).hex()
    os.makedirs(folder, exist_ok=True)
    try:
        with Image.open(f.stream) as im:
            im.convert("RGB").save(os.path.join(folder, token + ".png"))
    except OSError:
        abort(400, "that image couldn't be read")
    return token


def _ref_path(token: str | None) -> str | None:
    if not token:
        return None
    if not _REF.match(token):
        abort(400, "bad reference")
    p = colormatch._root("_refs", token + ".png")
    if not os.path.exists(p):
        abort(404, "that reference image is gone; upload it again")
    return p


@bp.route("/api/colormatch")
def api_colormatch():
    return jsonify(sessions=colormatch.list_sessions(), methods=colormatch.METHODS, modes=colormatch.MODES,
                   active=[j for j in jobs.active_for("colormatch")])


@bp.route("/api/colormatch/reference", methods=["POST"])
def api_colormatch_reference():
    token = _save_upload_image(request.files.get("image"), colormatch._root("_refs"))
    return jsonify(token=token)


@bp.route("/media/colormatch/ref/<token>")
def media_colormatch_ref(token):
    return send_file(_ref_path(token), max_age=3600)


@bp.route("/api/colormatch/analyze", methods=["POST"])
def api_colormatch_analyze():
    d = _body()
    ids = d.get("ids") or []
    if not ids:
        abort(400, "add at least one segment")
    vids = [_vid(i) for i in ids]
    method, mode = d.get("method", "exact"), d.get("mode", "seam")
    if method not in colormatch.METHODS or mode not in colormatch.MODES:
        abort(400, "unknown method or mode")
    fps = str(d.get("fps") or "auto").strip()
    if not re.match(r"^(auto|first|\d{1,3}(\.\d+)?|\d{1,6}/\d{1,6})$", fps):
        abort(400, "frame rate must be auto, first, or a number like 16 or 30000/1001")
    ovl = d.get("overlaps")
    if ovl is not None:
        try:
            ovl = [None if v in (None, "") else int(v) for v in ovl]
        except (TypeError, ValueError):
            abort(400, "overlaps must be whole numbers")
    sid = d.get("session")
    if sid and not colormatch.SID_RE.match(sid):
        abort(400, "bad session id")
    ref = _ref_path(d.get("reference"))
    fit_frames = _num(d, "fit_frames", int) or 5
    strength = _num(d, "strength")
    ends = d.get("ends")
    if ends is not None:
        try:
            ends = [None if v in (None, "", 0) else int(v) for v in ends]
        except (TypeError, ValueError):
            abort(400, "end frames must be whole numbers")
        if any(e is not None and e < 2 for e in ends):
            abort(400, "an end frame must be 2 or more")
    return _job(jobs.submit(
        "colormatch", f"Color match: analyse {len(vids)} segments", colormatch.analyze,
        [v["path"] for v in vids], reference=ref, method=method, fit_frames=fit_frames, mode=mode,
        strength=strength, luma_strength=_num(d, "luma_strength"), color_strength=_num(d, "color_strength"),
        fps=fps, ends=ends, overlaps=ovl, names=[v["name"] for v in vids], session=sid, ids=ids, ref="colormatch"))


@bp.route("/api/colormatch/<sid>", methods=["GET", "DELETE"])
def api_colormatch_session(sid):
    if request.method == "DELETE":
        _cm(colormatch.delete, sid)
        return jsonify(ok=True)
    return jsonify(_cm(colormatch.load, sid))


@bp.route("/api/colormatch/<sid>/preview")
def api_colormatch_preview(sid):
    seg = _num(request.args, "seg", int) or 0
    frame = _num(request.args, "frame", int) or 0
    return Response(_cm(colormatch.preview, sid, seg, frame), mimetype="image/png",
                    headers={"Cache-Control": "no-store"})


@bp.route("/api/colormatch/<sid>/render", methods=["POST"])
def api_colormatch_render(sid):
    _cm(colormatch.load, sid)
    d = _body()
    if jobs.active_for(f"colormatch:{sid}"):
        abort(409, "this session is already rendering")
    return _job(jobs.submit(
        "colormatch", "Color match: render", colormatch.render, sid, lossless=bool(d.get("lossless")),
        crossfade=_num(d, "crossfade", int) or 0, keep_frames=bool(d.get("keep_frames")),
        segment_clips=bool(d.get("segment_clips")), ref=f"colormatch:{sid}"))


@bp.route("/api/colormatch/<sid>/file/<path:name>")
def api_colormatch_file(sid, name):
    p = _cm(colormatch.output_path, sid, name)
    if not os.path.isfile(p):
        abort(404)
    return send_file(p, conditional=True, as_attachment=request.args.get("dl") == "1")


@bp.route("/api/colormatch/image", methods=["POST"])
def api_colormatch_image():
    """Correct one image (e.g. a handoff frame) against a reference; returns the PNG."""
    folder = colormatch._root("_images")
    src = _save_upload_image(request.files.get("image"), folder)
    if request.files.get("reference"):
        ref = os.path.join(folder, _save_upload_image(request.files["reference"], folder) + ".png")
    else:
        ref = _ref_path(request.form.get("reference")) or abort(400, "a reference image is needed")
    method = request.form.get("method", "exact")
    if method not in colormatch.METHODS:
        abort(400, "unknown method")
    strength = _num(request.form, "strength")
    lum, col = _num(request.form, "luma_strength"), _num(request.form, "color_strength")
    if lum is not None or col is not None:
        base = 1.0 if strength is None else strength
        strength = (base if lum is None else lum, base if col is None else col)
    out = os.path.join(folder, src + "-matched.png")
    colormatch.correct_image(os.path.join(folder, src + ".png"), ref, out, method,
                             1.0 if strength is None else strength)
    stem = secure_filename(os.path.splitext(request.files["image"].filename or "image")[0]) or "image"
    return send_file(out, mimetype="image/png", as_attachment=True, download_name=f"{stem}-matched.png")


# --- lineage -------------------------------------------------------------------------------------

def _lin(fn, *a, **k):
    try:
        return fn(*a, **k)
    except ValueError as e:
        abort(400, str(e))
    except LookupError as e:
        abort(404, str(e))


def _scope() -> str | None:
    s = (request.args.get("scope") or _body().get("scope") or "").strip()
    return s or None


@bp.route("/api/lineage")
def api_lineage():
    scope = _scope()
    return jsonify(folders=lineage.folders(), env_folders=lineage.env_folders(), scope=scope,
                   chains=_lin(lineage.chains, scope),
                   backends=features.backends(),
                   active=[j for j in jobs.active_for("lineage")])


@bp.route("/api/lineage/folders", methods=["POST"])
def api_lineage_folders():
    d = _body()
    path = (d.get("path") or "").strip()
    if not path:
        abort(400, "a folder path is needed")
    if d.get("action") == "remove":
        return jsonify(folders=_lin(lineage.remove_folder, path))
    return jsonify(folders=_lin(lineage.add_folder, path))


@bp.route("/api/lineage/scan", methods=["POST"])
def api_lineage_scan():
    d = _body()
    only = d.get("folder")
    if only:
        only = os.path.abspath(only.strip().strip('"'))
        if not os.path.isdir(only):
            abort(400, f"no such folder: {only}")
    elif not lineage.folders():
        abort(400, "add a folder first")
    if jobs.active_for("lineage"):
        abort(409, "a lineage scan is already running")
    label = f"Lineage: index {os.path.basename(only) if only else 'all folders'}"
    return _job(jobs.submit("lineage", label, lineage.scan, only=[only] if only else None, ref="lineage"))


@bp.route("/api/lineage/chain/<int:number>")
def api_lineage_chain(number):
    return jsonify(_lin(lineage.chain, number, _scope()))


@bp.route("/api/lineage/chain/<int:number>/pick", methods=["POST"])
def api_lineage_pick(number):
    tip = _num(_body(), "tip", int)
    return jsonify(_lin(lineage.pick, number, tip, _scope()))


@bp.route("/api/lineage/mark", methods=["POST"])
def api_lineage_mark():
    d = _body()
    _lin(lineage.mark, _num(d, "clip", int), d.get("mark") or None)
    return jsonify(ok=True)


@bp.route("/api/lineage/link", methods=["POST"])
def api_lineage_link():
    d = _body()
    child = _num(d, "child", int)
    parent = d.get("parent")
    if isinstance(parent, str) and parent.strip() and not parent.strip().isdigit():
        parent = lineage.resolve(parent, _scope())  # a label or file name
        if parent is None:
            abort(404, "no clip matches that label or name")
    parent = int(parent) if parent not in (None, "") else None
    _lin(lineage.override, child, parent, d.get("kind", "link"))
    return jsonify(ok=True)


@bp.route("/api/lineage/faces", methods=["POST"])
def api_lineage_faces():
    """Find faces for one chain (number) or every chain in the view, then score them."""
    from .. import faces
    if not faces.available():
        abort(400, "face checks need insightface: see requirements-faces.txt")
    d = _body()
    number = _num(d, "chain", int)
    if jobs.active_for("lineage-faces"):
        abort(409, "a face check is already running")
    label = f"Lineage: faces in c{number:03d}" if number is not None else "Lineage: faces in every chain"
    return _job(jobs.submit("lineage", label, lineage.check_faces, number=number, scope=_scope(),
                            redo=bool(d.get("redo")), ref="lineage-faces"))


@bp.route("/api/lineage/clip/<int:cid>/end", methods=["POST"])
def api_lineage_end(cid):
    """Set where a take ends (frame number, 1 = first); empty = its last frame again."""
    return jsonify(_lin(lineage.set_end, cid, _num(_body(), "end", int)))


@bp.route("/api/lineage/clip/<int:cid>/face/<int:n>")
def api_lineage_frame_face(cid, n):
    return jsonify(_lin(lineage.frame_face, cid, n, _scope()))


@bp.route("/api/lineage/clip/<int:cid>/export", methods=["POST"])
def api_lineage_export(cid):
    """Save one frame as a full-size PNG handoff (for Qwen or the next generation)."""
    n = _num(_body(), "frame", int)
    if not n:
        abort(400, "which frame?")
    p = _lin(lineage.export_frame, cid, n, _scope())
    return jsonify(path=p, name=os.path.basename(p))


@bp.route("/media/lineage/<int:cid>/frame/<int:n>")
def media_lineage_frame(cid, n):
    from io import BytesIO
    from PIL import Image
    h = _num(request.args, "h", int) or 480
    img = _lin(lineage.grab, cid, n, max(120, min(h, 2160)))
    buf = BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=88)
    return Response(buf.getvalue(), mimetype="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@bp.route("/api/lineage/suggest/<int:cid>")
def api_lineage_suggest(cid):
    return jsonify(_lin(lineage.suggest, cid, _scope(), request.args.get("backend", "clip")))


@bp.route("/api/lineage/publish", methods=["POST"])
def api_lineage_publish():
    """Put a chain's clips in the library (in place, under their folder) for Color match."""
    ids = []
    ends = _body().get("ends") or []
    for cid in _body().get("clips") or []:
        c = _lin(lineage.clip, int(cid))
        if not os.path.exists(c["path"]):
            abort(404, f"missing file: {c['path']}")
        ids.append(library.add_path(c["path"], root=c["folder"])["id"])
    if not ids:
        abort(400, "no clips selected")
    return jsonify(ids=ids, ends=[e if isinstance(e, int) and e > 0 else None for e in ends][:len(ids)])


@bp.route("/media/lineage/<int:cid>/<which>")
def media_lineage(cid, which):
    if which in ("first", "last"):
        p = lineage.thumb_path(cid, which)
        if not os.path.exists(p):
            abort(404)
        return send_file(p, max_age=86400)
    if which == "video":
        c = _lin(lineage.clip, cid)
        if not os.path.exists(c["path"]):
            abort(404)
        return send_file(c["path"], conditional=True)
    abort(404)
