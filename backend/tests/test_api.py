"""API integration tests using FastAPI TestClient.

The real AI pipeline (ChangeFormer / SegFormer / YOLO) is NOT loaded:
the app lifespan is replaced with a no-op and a fake pipeline object is
injected into app.state. A temporary SQLite file replaces the production
database via dependency_overrides.
"""

import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import AnalysisHistory

SAMPLE_RESULT = {
    "success": True,
    "image_name": "test",
    "image_size": {"width": 16, "height": 16},
    "processing_time": 0.01,
    "changed_pixels": 100,
    "change_percentage": 39.06,
    "objects": [
        {
            "id": 1,
            "class_id": 2,
            "class_name": "building",
            "confidence": 0.9,
            "bbox": [2, 2, 6, 6],
            "centroid": [5, 5],
            "polygon": None,
            "change_overlap": 1.0,
            "status": "appeared",
        }
    ],
    "statistics": {
        "summary": {"total_changed_pixels": 100, "total_transitions": 1, "major_change": "Background -> Building"},
        "transitions": [],
    },
    "severity": {"low_pixels": 100, "medium_pixels": 0, "high_pixels": 0, "mean_confidence": 0.5},
    "report": {"change_detection": {"change_percentage": 39.06}},
    "files": {"overlay": "/static/outputs/test_overlay.png", "json": "/static/outputs/test.json"},
}


class FakePipeline:
    def __init__(self, result=None, error=None):
        self.result = result or SAMPLE_RESULT
        self.error = error

    def run(self, *args, **kwargs):
        if self.error:
            raise RuntimeError(self.error)
        return dict(self.result)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # No-op lifespan: skip loading the real AI models.
    @asynccontextmanager
    async def dummy_lifespan(app):
        yield

    app.router.lifespan_context = dummy_lifespan
    app.state.pipeline = None

    # Point the route file folders at a temp dir so tests never touch real
    # uploads/outputs.
    import app.routes as routes

    monkeypatch.setattr(routes, "UPLOAD_FOLDER", tmp_path / "uploads")
    monkeypatch.setattr(routes, "OUTPUT_FOLDER", tmp_path / "outputs")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "outputs").mkdir()

    # Temp SQLite database for history rows.
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        c.testing_session = TestingSessionLocal
        yield c

    app.dependency_overrides.clear()
    app.state.pipeline = None


def _tiny_png(path):
    img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _poll_result(client, url):
    for _ in range(100):
        response = client.get(url)
        data = response.json()
        if data.get("status") in ("completed", "error"):
            return data
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_pipeline_not_loaded(client):
    app.state.pipeline = None
    response = client.get("/pipeline")
    assert response.json() == {"pipeline": "not loaded"}


def test_pipeline_loaded(client):
    app.state.pipeline = FakePipeline()
    response = client.get("/pipeline")
    assert response.status_code == 200
    assert response.json()["pipeline"] == "loaded"


def test_stats_empty(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["totals"]["analyses"] == 0


def test_history_empty(client):
    response = client.get("/history")
    assert response.status_code == 200
    assert response.json() == []


def test_predict_503_without_pipeline(client, tmp_path):
    app.state.pipeline = None
    before = tmp_path / "b.png"
    after = tmp_path / "a.png"
    _tiny_png(before)
    _tiny_png(after)
    response = client.post(
        "/predict",
        files={"before": ("b.png", before.read_bytes(), "image/png"),
               "after": ("a.png", after.read_bytes(), "image/png")},
    )
    assert response.status_code == 503


def test_predict_full_flow(client, tmp_path):
    app.state.pipeline = FakePipeline()
    before = tmp_path / "b.png"
    after = tmp_path / "a.png"
    _tiny_png(before)
    _tiny_png(after)

    response = client.post(
        "/predict",
        files={"before": ("b.png", before.read_bytes(), "image/png"),
               "after": ("a.png", after.read_bytes(), "image/png")},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    result = _poll_result(client, f"/predict-result/{job_id}")
    assert result["status"] == "completed"
    assert result["change_detection"]["changed_pixels"] == 100
    assert result["objects_detected"] == 1
    assert result["history_id"] is not None

    # History row was persisted.
    history = client.get("/history").json()
    assert len(history) == 1
    assert history[0]["image_name"] == job_id
    assert history[0]["source"] == "upload"

    # Polling again must be idempotent: the history row is reused, not
    # re-inserted (regression: UNIQUE constraint on image_name).
    again = _poll_result(client, f"/predict-result/{job_id}")
    assert again["status"] == "completed"
    assert again["history_id"] == result["history_id"]
    assert len(client.get("/history").json()) == 1


def test_predict_error_reported(client, tmp_path):
    app.state.pipeline = FakePipeline(error="boom")
    before = tmp_path / "b.png"
    after = tmp_path / "a.png"
    _tiny_png(before)
    _tiny_png(after)

    response = client.post(
        "/predict",
        files={"before": ("b.png", before.read_bytes(), "image/png"),
               "after": ("a.png", after.read_bytes(), "image/png")},
    )
    job_id = response.json()["job_id"]
    result = _poll_result(client, f"/predict-result/{job_id}")
    assert result["status"] == "error"
    assert "boom" in result["detail"]


def test_predict_result_unknown_job_404(client):
    assert client.get("/predict-result/does-not-exist").status_code == 404
    assert client.get("/progress/does-not-exist").status_code == 404


def test_predict_result_queued_status(client):
    from app.jobs import store

    store.create("queued_job_1", "upload", None)
    store.queue("queued_job_1")
    response = client.get("/predict-result/queued_job_1")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert "stage" in data


def test_predict_result_interrupted_status(client):
    from app.jobs import store

    store.create("interrupted_job_1", "upload", None)
    store.mark_interrupted()
    response = client.get("/predict-result/interrupted_job_1")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert "restart" in data["detail"]


def test_download_outputs(client):
    # Create a fake output file the download endpoint can zip up.
    import app.routes as routes

    out = routes.OUTPUT_FOLDER / "img_test_overlay.png"
    out.write_bytes(b"fake-image-bytes")

    response = client.get("/download/img_test")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert "img_test_overlay.png" in zf.namelist()


def test_download_missing_404(client):
    assert client.get("/download/img_missing").status_code == 404


def test_download_rejects_path_traversal(client):
    # Regression: image_name was previously interpolated straight into
    # OUTPUT_FOLDER.glob(f"{image_name}*"), so %2F-decoded ".." segments
    # could read arbitrary files outside the static folders.  Starlette
    # refuses ".." at routing time (404) and the regex guard rejects any
    # remaining malformed name (400) — either way nothing is served.
    assert client.get("/download/..%2F..%2Fapp%2Froutes").status_code in (400, 404)
    assert client.get("/download/../../app/routes").status_code in (400, 404)
    assert client.get("/download/..%5C..%5Capp").status_code in (400, 404)
    assert client.get("/download/UPPER_CASE").status_code in (400, 404)
    assert client.get("/download/img_ab%2Fcd").status_code in (400, 404)


def test_map_predict_rejects_invalid_input(client):
    bad_payloads = [
        {"lat": 91, "lng": 0, "buffer_deg": 0.01},          # lat out of range
        {"lat": 33, "lng": 0, "buffer_deg": 0},              # zero buffer
        {"lat": 33, "lng": 0, "buffer_deg": -1},             # negative buffer
        {"lat": "abc", "lng": 0, "buffer_deg": 0.01},        # non-numeric lat
        {"lat": 33, "lng": 0, "buffer_deg": 0.01, "date1": "not-a-date"},
        {"lat": 33, "lng": 0, "buffer_deg": 0.01, "date1": "2024-06-01", "date2": "2024-01-01"},
        {},                                                  # missing lat/lng
    ]
    for payload in bad_payloads:
        response = client.post("/map-predict", json=payload)
        assert response.status_code == 400, f"expected 400 for {payload}"


def test_delete_history(client):
    db = client.testing_session()
    record = AnalysisHistory(
        report_name="Test",
        image_name="img_delete_me",
        source="upload",
        changed_pixels=1,
        change_percentage=1.0,
        processing_time=1.0,
        object_count=0,
    )
    db.add(record)
    db.commit()
    record_id = record.id
    db.close()

    response = client.delete(f"/history/{record_id}")
    assert response.status_code == 200
    assert response.json()["deleted_id"] == record_id

    assert client.delete(f"/history/{record_id}").status_code == 404