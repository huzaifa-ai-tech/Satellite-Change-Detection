"""Job progress helpers — thin wrappers over the persistent SQLite store.

The public functions keep their historical signatures so the pipeline
(``src/pipeline.py``) and tests work unchanged; the backing storage is
now ``app.jobs.JobStore`` instead of an in-memory dict.
"""

from app.jobs import JOB_TTL_SECONDS, store


def create_job(job_id: str, kind: str = "generic", payload=None):
    store.create(job_id, kind, payload)


def update_job(job_id: str, progress: int, stage: str):
    store.update(job_id, progress, stage)


def get_job(job_id: str):
    return store.get(job_id)