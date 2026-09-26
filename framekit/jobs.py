"""In-process background jobs with progress, shared by every tool.

    job = jobs.submit("frames", "Extract frames from x.mp4", fn, arg1, kw=...)

`fn(job, *args, **kwargs)` runs on a worker thread. It reports through
job.update(progress=, total=, message=) and calls job.check() in loops so
cancellation takes effect. Its return value becomes job.result.
At most MAX_RUNNING jobs execute at once; the rest wait as "queued".
"""
from __future__ import annotations

import secrets
import sys
import threading
import time
import traceback

MAX_RUNNING = 2
KEEP = 200  # finished jobs remembered for the UI

_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()
_slots = threading.Semaphore(MAX_RUNNING)


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, kind: str, label: str, ref: str | None = None):
        self.id = secrets.token_hex(4)
        self.kind, self.label, self.ref = kind, label, ref
        self.state = "queued"
        self.progress, self.total = 0, 0
        self.message = ""
        self.error = None
        self.result = None
        self.created, self.started, self.finished = time.time(), None, None
        self._cancel = threading.Event()

    def update(self, **kw) -> None:
        for k, v in kw.items():
            setattr(self, k, v)

    def check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "kind", "label", "ref", "state", "progress", "total", "message",
            "error", "result", "created", "started", "finished")}


def _run(job: Job, fn, args, kwargs) -> None:
    with _slots:
        if job.cancelled:
            job.update(state="cancelled", finished=time.time())
            return
        job.update(state="running", started=time.time())
        try:
            job.result = fn(job, *args, **kwargs)
            job.update(state="done")
        except Cancelled:
            job.update(state="cancelled")
        except (ValueError, LookupError) as e:  # bad input or missing file: a message is enough
            job.update(state="error", error=str(e))
        except Exception as e:  # surfaced to the UI; never kills the server
            job.update(state="error", error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            job.finished = time.time()


def submit(kind: str, label: str, fn, *args, ref: str | None = None, **kwargs) -> Job:
    job = Job(kind, label, ref)
    with _lock:
        _jobs[job.id] = job
        done = sorted((j for j in _jobs.values() if j.finished), key=lambda j: j.finished)
        for old in done[:-KEEP]:
            _jobs.pop(old.id, None)
    threading.Thread(target=_run, args=(job, fn, args, kwargs), daemon=True).start()
    return job


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    with _lock:
        js = list(_jobs.values())
    return [j.to_dict() for j in sorted(js, key=lambda j: j.created, reverse=True)]


def active_for(ref: str) -> list[dict]:
    return [j for j in list_jobs() if j["ref"] == ref and j["state"] in ("queued", "running")]


def cancel(job_id: str) -> bool:
    j = get(job_id)
    if not j or j.finished:
        return False
    j._cancel.set()
    return True


def run_sync(fn, *args, label: str = "", verbose: bool = True, **kwargs):
    """Run a job function inline (CLI), printing progress to stderr."""
    job = Job("cli", label)
    job.state = "running"
    stop = threading.Event()
    tty = verbose and sys.stderr.isatty()

    def ticker():
        while not stop.wait(0.5):
            line = f"{label}: {job.message} {job.progress}/{job.total or '?'}"
            print("\r" + line.ljust(100)[:100], end="", file=sys.stderr, flush=True)
    t = threading.Thread(target=ticker, daemon=True)
    if tty:
        t.start()
    try:
        return fn(job, *args, **kwargs)
    finally:
        stop.set()
        if tty:
            t.join()
            print("\r" + " " * 100 + "\r", end="", file=sys.stderr, flush=True)
