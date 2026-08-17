"""Tests for the persistent SQLite job store (app.jobs)."""

import numpy as np
import pytest

from app.jobs import JobStore, json_safe


@pytest.fixture()
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.db")
    yield s
    s.close()


def test_crud_roundtrip(store):
    store.create("job_a", "upload", {"image_name": "img_x"})
    job = store.get("job_a")
    assert job["status"] == "running"
    assert job["progress"] == 0

    store.update("job_a", 40, "Working")
    assert store.get("job_a")["progress"] == 40

    store.finish("job_a", {"ok": True, "count": 3})
    assert store.get("job_a")["status"] == "completed"
    assert store.get_result("job_a") == {"ok": True, "count": 3}


def test_result_missing_while_running(store):
    store.create("job_b")
    assert store.get_result("job_b") is None
    store.finish("job_b", None)
    assert store.get_result("job_b") is None  # completed without payload


def test_error_state(store):
    store.create("job_c")
    store.error("job_c", "exploded")
    job = store.get("job_c")
    assert job["status"] == "error"
    assert job["error"] == "exploded"
    assert store.get_result("job_c") is None


def test_mark_interrupted(store):
    store.create("job_d")
    store.create("job_e")
    store.finish("job_d", {"done": True})
    store.mark_interrupted()
    assert store.get("job_d")["status"] == "completed"
    assert store.get("job_e")["status"] == "interrupted"


def test_queue_status(store):
    store.create("job_q")
    store.queue("job_q")
    job = store.get("job_q")
    assert job["status"] == "queued"
    assert store.get_result("job_q") is None


def test_mark_interrupted_covers_queued(store):
    store.create("job_r")
    store.queue("job_r")
    store.mark_interrupted()
    assert store.get("job_r")["status"] == "interrupted"


def test_worker_pool_serializes_jobs():
    import threading
    import time

    from app.jobs import WorkerPool

    pool = WorkerPool(max_workers=1)
    lock = threading.Lock()
    active = {"count": 0, "max": 0}

    def job():
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.1)
        with lock:
            active["count"] -= 1

    futures = [pool.submit(job) for _ in range(3)]
    for future in futures:
        future.result(timeout=10)
    pool.shutdown()
    assert active["max"] == 1, "more than one job ran concurrently"


def test_submit_job_completes():
    import time

    from app.jobs import store, submit_job

    store.create("job_s")
    submit_job("job_s", lambda: {"done": True, "n": 5})
    for _ in range(200):
        if store.get("job_s")["status"] == "completed":
            break
        time.sleep(0.05)
    assert store.get("job_s")["status"] == "completed"
    assert store.get_result("job_s") == {"done": True, "n": 5}


def test_submit_job_reports_error():
    import time

    from app.jobs import store, submit_job

    store.create("job_t")

    def boom():
        raise RuntimeError("exploded")

    submit_job("job_t", boom)
    for _ in range(200):
        if store.get("job_t")["status"] in ("completed", "error"):
            break
        time.sleep(0.05)
    job = store.get("job_t")
    assert job["status"] == "error"
    assert "exploded" in (job.get("error") or "")


def test_ttl_expiry(store, monkeypatch):
    fake_now = {"t": 1_000_000.0}
    monkeypatch.setattr("app.jobs.time.time", lambda: fake_now["t"])

    store.create("job_ttl")
    store.finish("job_ttl", {"v": 1})
    assert store.get_result("job_ttl") == {"v": 1}

    fake_now["t"] += 24 * 60 * 60 + 1
    assert store.get_result("job_ttl") is None
    assert store.get("job_ttl") is None  # pruned


def test_completed_result_survives_job_ttl(store, monkeypatch):
    # Regression: completed results were pruned after the 1h job TTL even
    # though RESULT_TTL_SECONDS (24h) is meant to keep them reachable.
    fake_now = {"t": 1_000_000.0}
    monkeypatch.setattr("app.jobs.time.time", lambda: fake_now["t"])

    store.create("job_keep")
    store.finish("job_keep", {"v": 1})
    fake_now["t"] += 60 * 60 + 1  # past JOB_TTL_SECONDS
    assert store.get("job_keep")["status"] == "completed"
    assert store.get_result("job_keep") == {"v": 1}


def test_running_and_queued_jobs_never_pruned(store, monkeypatch):
    fake_now = {"t": 1_000_000.0}
    monkeypatch.setattr("app.jobs.time.time", lambda: fake_now["t"])

    store.create("job_run")
    store.update("job_run", 50, "Working")
    store.create("job_wait")
    store.queue("job_wait")
    fake_now["t"] += 3 * 60 * 60  # far past JOB_TTL_SECONDS
    assert store.get("job_run") is not None
    assert store.get("job_wait") is not None


def test_error_job_pruned_after_ttl(store, monkeypatch):
    fake_now = {"t": 1_000_000.0}
    monkeypatch.setattr("app.jobs.time.time", lambda: fake_now["t"])

    store.create("job_err")
    store.error("job_err", "boom")
    fake_now["t"] += 60 * 60 + 1
    assert store.get("job_err") is None


def test_create_replaces_previous_state(store):
    store.create("job_f", "upload", {"a": 1})
    store.update("job_f", 70, "Almost done")
    store.create("job_f", "map", {"b": 2})
    job = store.get("job_f")
    assert job["progress"] == 0
    assert job["stage"] == "Starting"


def test_json_safe_numpy_and_path(tmp_path):
    from pathlib import Path

    value = {
        "n": np.int64(3),
        "f": np.float32(1.5),
        "arr": [np.int32(1), np.int64(2)],
        "path": Path("C:/x/y.png"),
        "plain": "text",
        "none": None,
    }
    safe = json_safe(value)
    assert safe["n"] == 3
    assert safe["f"] == 1.5
    assert safe["arr"] == [1, 2]
    assert safe["plain"] == "text"
    assert safe["none"] is None
    assert safe["path"] == str(Path("C:/x/y.png"))
    import json
    json.dumps(safe)  # fully serializable