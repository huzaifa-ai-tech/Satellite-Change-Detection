"""Tests for the persistent job store (app.progress / app.jobs)."""

import time as real_time

import app.jobs
import app.progress as progress


def test_create_job():
    progress.create_job("job_1")
    job = progress.get_job("job_1")
    assert job["progress"] == 0
    assert job["status"] == "running"
    assert job["stage"] == "Starting"


def test_update_job():
    progress.create_job("job_2")
    progress.update_job("job_2", 50, "Analyzing")
    job = progress.get_job("job_2")
    assert job["progress"] == 50
    assert job["stage"] == "Analyzing"


def test_update_unknown_job_is_safe():
    progress.update_job("missing", 50, "Analyzing")
    assert progress.get_job("missing") is None


def test_get_unknown_job_returns_none():
    assert progress.get_job("does_not_exist") is None


def test_running_job_never_expires(monkeypatch):
    # Regression: the old TTL pruned *running* jobs past JOB_TTL_SECONDS,
    # which deleted in-flight jobs (and their results) mid-run.  Only
    # errored/interrupted rows are pruned; running rows are never expired.
    fake_now = {"t": 1_000_000.0}

    def fake_time():
        return fake_now["t"]

    monkeypatch.setattr(app.jobs.time, "time", fake_time)

    progress.create_job("job_run")
    assert progress.get_job("job_run") is not None

    fake_now["t"] += progress.JOB_TTL_SECONDS * 3
    assert progress.get_job("job_run") is not None


def test_error_job_expires_after_ttl(monkeypatch):
    fake_now = {"t": 1_000_000.0}

    def fake_time():
        return fake_now["t"]

    monkeypatch.setattr(app.jobs.time, "time", fake_time)

    progress.create_job("job_err")
    app.jobs.store.error("job_err", "boom")
    assert progress.get_job("job_err") is not None

    fake_now["t"] += progress.JOB_TTL_SECONDS + 1
    assert progress.get_job("job_err") is None

    real_time.sleep(0.01)  # keep import referenced