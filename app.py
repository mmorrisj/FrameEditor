"""Flask app: FrameEditor — extract a video's frames, edit regions, repackage.

Thin HTTP layer only; ffmpeg plumbing lives in video.py, Pillow ops in edits.py.
"""
import os
import re

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

import edits
import video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024**3  # uploads up to 4 GB

_PID = re.compile(r"^[a-z0-9-]{4,60}$")
MAX_RANGE = 2000  # frames per edit request; guards a typo'd range from minutes of Pillow work


def _dir(pid: str) -> str:
    if not _PID.match(pid):
        abort(400)
    d = video.project_dir(pid)
    if not os.path.isdir(d):
        abort(404)
    return d


@app.route("/")
def index():
    return render_template("index.html")


# --- projects ---------------------------------------------------------------

@app.route("/api/projects", methods=["GET"])
def projects():
    return jsonify(video.list_projects())


@app.route("/api/projects", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        abort(400, "no video file in request")
    try:
        pid = video.create_project(f)
    except ValueError as e:
        abort(400, str(e))
    video.start_extract(pid)
    return jsonify({"id": pid})


@app.route("/api/projects/<pid>", methods=["DELETE"])
def delete(pid):
    _dir(pid)
    video.delete_project(pid)
    return jsonify({"ok": True})


@app.route("/api/projects/<pid>/status")
def status(pid):
    _dir(pid)
    return jsonify(video.status(pid))


# --- media ------------------------------------------------------------------

@app.route("/media/<pid>/frame/<int:n>")
def frame(pid, n):
    return send_from_directory(os.path.join(_dir(pid), "frames"), f"f{n:06d}.png")


@app.route("/media/<pid>/thumb/<int:n>")
def thumb(pid, n):
    return send_from_directory(os.path.join(_dir(pid), "thumbs"), f"t{n:06d}.jpg")


@app.route("/api/projects/<pid>/frame/<int:n>/download")
def download_frame(pid, n):
    return send_from_directory(
        os.path.join(_dir(pid), "frames"), f"f{n:06d}.png",
        as_attachment=True, download_name=f"{pid}-frame{n:06d}.png")


@app.route("/api/projects/<pid>/output")
def output(pid):
    return send_from_directory(
        _dir(pid), "output.mp4", as_attachment=True, download_name=f"{pid}-edited.mp4")


# --- editing ----------------------------------------------------------------

@app.route("/api/projects/<pid>/edit", methods=["POST"])
def edit(pid):
    d = _dir(pid)
    data = request.get_json(force=True)
    total = video.meta(pid)["frames"]
    try:
        f0, f1 = int(data["from"]), int(data["to"])
        rect = {k: float(data["rect"][k]) for k in ("x", "y", "w", "h")}
        op = data["op"]
        strength = max(1, min(100, int(data.get("strength", 12))))
    except (KeyError, TypeError, ValueError):
        abort(400, "bad edit payload")
    if not (1 <= f0 <= f1 <= total):
        abort(400, f"frame range must be within 1..{total}")
    if f1 - f0 + 1 > MAX_RANGE:
        abort(400, f"range too large (max {MAX_RANGE} frames)")
    try:
        for n in range(f0, f1 + 1):  # ascending: clone cascades cleaned pixels
            edits.apply(d, n, rect, op, strength, total)
    except ValueError as e:
        abort(400, str(e))
    return jsonify(video.status(pid))


@app.route("/api/projects/<pid>/revert", methods=["POST"])
def revert(pid):
    d = _dir(pid)
    data = request.get_json(force=True)
    edits.revert(d, int(data["frame"]))
    return jsonify(video.status(pid))


@app.route("/api/projects/<pid>/frame/<int:n>/replace", methods=["POST"])
def replace(pid, n):
    d = _dir(pid)
    f = request.files.get("image")
    if not f or not f.filename:
        abort(400, "no image file in request")
    m = video.meta(pid)
    if not 1 <= n <= m["frames"]:
        abort(400)
    edits.replace(d, n, f, (m["width"], m["height"]))
    return jsonify(video.status(pid))


@app.route("/api/projects/<pid>/repackage", methods=["POST"])
def repackage(pid):
    _dir(pid)
    video.start_repackage(pid)
    return jsonify(video.status(pid))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8082)
