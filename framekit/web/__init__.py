"""Flask app: one page per tool over the shared library, plus JSON APIs.

Thin layer only: every route validates input and calls a tool function, or
submits it as a background job and returns the job id for the page to poll.
"""
from __future__ import annotations

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from .. import config


def create_app() -> Flask:
    config.ensure_dirs()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024**3  # uploads up to 8 GB

    from . import routes, editor_routes
    app.register_blueprint(routes.bp)
    app.register_blueprint(editor_routes.bp, url_prefix="/editor")

    from ..tools import unique
    unique.watcher()  # drop-folder watcher; idles while watching is switched off

    @app.errorhandler(ValueError)
    def bad(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(LookupError)
    def missing(e):
        if isinstance(e, (KeyError, IndexError)):  # programming errors, not "not found"
            raise e
        return jsonify(error=str(e)), 404

    @app.errorhandler(HTTPException)
    def http(e):
        return jsonify(error=e.description or e.name), e.code

    return app
