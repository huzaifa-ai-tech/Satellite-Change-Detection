"""Persistent SQLite job store and background worker helpers.

Job state (progress, stage, status, result payload) lives in SQLite at
``JOBS_DB_PATH`` (default ``backend/jobs.db``) instead of in memory, so
in-flight jobs and their results survive backend restarts.  Jobs left
"running" when the server went down are marked "interrupted" on the next
startup (see ``app/main.py`` lifespan).
"""

import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
JOBS_DB_PATH = os.getenv("JOBS_DB_PATH", str(BASE_DIR / "jobs.db"))

JOB_TTL_SECONDS = 60 * 60
RESULT_TTL_SECONDS = 24 * 60 * 60
# How many analyses may run at once. The pipeline loads every model into
# memory, so >1 worker on CPU (or without enough GPU VRAM) thrashes.
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    kind       TEXT NOT NULL DEFAULT 'generic',
    payload    TEXT,
    progress   INTEGER NOT NULL DEFAULT 0,
    stage      TEXT NOT NULL DEFAULT 'Starting',
    status     TEXT NOT NULL DEFAULT 'running',
    result     TEXT,
    error      TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
)
"""


def json_safe(value):
    """Recursively convert a result dict into JSON-serializable types.

    Handles dicts, lists/tuples, numpy scalars (via ``.item()``) and
    anything else by stringifying it.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


class JobStore:
    """Thread-safe SQLite-backed job store."""

    def __init__(self, path=JOBS_DB_PATH):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def _execute(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def create(self, job_id, kind="generic", payload=None):
        self._execute(
            "INSERT OR REPLACE INTO jobs (job_id, kind, payload, progress, stage, status, created_at, updated_at)"
            " VALUES (?, ?, ?, 0, 'Starting', 'running', ?, ?)",
            (job_id, kind, json.dumps(payload, default=str), time.time(), time.time()),
        )

    def update(self, job_id, progress, stage):
        self._execute(
            "UPDATE jobs SET progress = ?, stage = ?, updated_at = ? WHERE job_id = ?",
            (int(progress), str(stage), time.time(), job_id),
        )

    def finish(self, job_id, result=None):
        payload = json.dumps(json_safe(result), default=str) if result is not None else None
        self._execute(
            "UPDATE jobs SET status = 'completed', progress = 100, stage = 'Completed',"
            " result = ?, updated_at = ? WHERE job_id = ?",
            (payload, time.time(), job_id),
        )

    def error(self, job_id, message):
        self._execute(
            "UPDATE jobs SET status = 'error', stage = ?, error = ?, updated_at = ? WHERE job_id = ?",
            (str(message), str(message), time.time(), job_id),
        )

    def update_result(self, job_id, result):
        """Persist an updated result payload (e.g. after the history row is
        written), without touching status or progress."""
        payload = json.dumps(json_safe(result), default=str)
        self._execute(
            "UPDATE jobs SET result = ?, updated_at = ? WHERE job_id = ?",
            (payload, time.time(), job_id),
        )

    def get(self, job_id):
        """Return a light job dict (no result payload) or None if expired."""
        with self._lock:
            row = self._conn.execute(
                "SELECT job_id, progress, stage, status, error, updated_at FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        job_id, progress, stage, status, error, updated_at = row
        age = time.time() - updated_at
        # Completed results are kept for the (much longer) result TTL so a
        # finished report stays reachable; stale failed/interrupted jobs are
        # pruned after the job TTL. Running/queued rows are never pruned
        # here: their worker may still be executing (or they get marked
        # "interrupted" on the next startup), and deleting them mid-run
        # would silently lose the result when the worker finishes.
        if status == "completed" and age > RESULT_TTL_SECONDS:
            self._execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return None
        if status in ("error", "interrupted") and age > JOB_TTL_SECONDS:
            self._execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return None
        return {
            "job_id": job_id,
            "progress": progress,
            "stage": stage,
            "status": status,
            "error": error,
        }

    def get_result(self, job_id):
        """Return the parsed completed result, or None if not done/expired."""
        with self._lock:
            row = self._conn.execute(
                "SELECT status, result, updated_at FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        status, result, updated_at = row
        if status != "completed" or result is None:
            return None
        if time.time() - updated_at > RESULT_TTL_SECONDS:
            self._execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            return None
        try:
            return json.loads(result)
        except (TypeError, ValueError):
            logger.warning("Stored result for %s is not valid JSON", job_id)
            return None

    def mark_interrupted(self):
        """Mark jobs left running/queued (server crashed/restarted) as interrupted."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = 'interrupted', stage = 'Interrupted by restart',"
                " updated_at = ? WHERE status IN ('running', 'queued')",
                (time.time(),),
            )
            self._conn.commit()
            count = cur.rowcount
        if count:
            logger.info("Marked %d in-flight job(s) as interrupted after restart", count)

    def queue(self, job_id):
        """Mark an existing job as queued (waiting for a free worker)."""
        self._execute(
            "UPDATE jobs SET status = 'queued', updated_at = ? WHERE job_id = ?",
            (time.time(), job_id),
        )

    def close(self):
        with self._lock:
            self._conn.close()


store = JobStore()


class WorkerPool:
    """Bounded background executor for analysis jobs.

    A single pool is shared by the whole app so at most ``MAX_WORKERS``
    analyses run concurrently; the rest wait in the executor queue with
    status "queued".
    """

    def __init__(self, max_workers=MAX_WORKERS):
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="job-worker"
        )

    def submit(self, fn):
        return self._executor.submit(fn)

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


pool = WorkerPool()


def submit_job(job_id, fn):
    """Queue ``fn`` for background execution on the bounded worker pool.

    The job is marked "queued" until a worker picks it up; the worker then
    sets "running" and stores the result (or error) via the store.
    """
    store.queue(job_id)

    def _run():
        try:
            store.update(job_id, 1, "Started")
            result = fn()
            store.finish(job_id, result)
        except Exception as e:
            logger.exception("Job %s failed", job_id)
            store.error(job_id, str(e))

    pool.submit(_run)