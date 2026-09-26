"""Routes for the library and every suite tool except the editor."""
from __future__ import annotations

import os
import re

from flask import Blueprint, Response, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from .. import config, features, fileops, jobs, library, samples
from ..tools import analysis, audio, dupes, frames, grouping, unique

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


@bp.route("/dupes")
def page_dupes():
    return render_template("dupes.html", page="dupes")


@bp.route("/groups")
def page_groups():
    return render_template("groups.html", page="groups")


@bp.route("/unique")
def page_unique():
    return render_template("unique.html", page="unique")


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
