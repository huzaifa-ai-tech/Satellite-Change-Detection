import io
import logging
import re
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.jobs import store, submit_job
from app.models import AnalysisHistory
from app.progress import create_job, update_job, get_job
from src.config import Config

logger = logging.getLogger("satellite")

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = BASE_DIR / "app" / "static" / "uploads"
OUTPUT_FOLDER = BASE_DIR / "app" / "static" / "outputs"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Image names are generated server-side as ``img_<hex>`` / ``map_<hex>``;
# anything else (slashes, dots, uppercase, long strings) is rejected so the
# /download glob can never escape the static folders.
SAFE_IMAGE_NAME = re.compile(r"^[a-z0-9_]{1,64}$")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/pipeline")
def pipeline_status(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return {"pipeline": "not loaded"}
    return {
        "pipeline": "loaded",
        "config": Config.summary(),
    }


@router.get("/stats")
def get_stats(request: Request, db: Session = Depends(get_db)):
    records = db.query(AnalysisHistory).all()

    total = len(records)
    total_objects = sum(r.object_count or 0 for r in records)
    total_changed = sum(r.changed_pixels or 0 for r in records)
    avg_change = round(sum(r.change_percentage or 0 for r in records) / total, 2) if total else 0.0
    avg_time = round(sum(r.processing_time or 0 for r in records) / total, 2) if total else 0.0

    pipeline = getattr(request.app.state, "pipeline", None)
    recent = (
        db.query(AnalysisHistory)
        .order_by(AnalysisHistory.id.desc())
        .limit(6)
        .all()
    )

    return {
        "success": True,
        "totals": {
            "analyses": total,
            "objects": total_objects,
            "changed_pixels": total_changed,
            "average_change_percentage": avg_change,
            "average_processing_time": avg_time,
        },
        "pipeline": {
            "status": "loaded" if pipeline else "not loaded",
            "config": Config.summary() if pipeline else None,
            "device": str(Config.DEVICE) if pipeline else None,
        },
        "recent": [
            {
                "id": r.id,
                "report_name": r.report_name,
                "image_name": r.image_name,
                "source": r.source,
                "change_percentage": r.change_percentage,
                "object_count": r.object_count,
                "overlay_path": r.overlay_path,
                "created_at": str(r.created_at),
            }
            for r in recent
        ],
    }


@router.get("/history")
def get_history(db: Session = Depends(get_db)):
    records = (
        db.query(AnalysisHistory)
        .order_by(AnalysisHistory.id.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "report_name": r.report_name,
            "image_name": r.image_name,
            "source": r.source,
            "changed_pixels": r.changed_pixels,
            "change_percentage": r.change_percentage,
            "processing_time": r.processing_time,
            "object_count": r.object_count,
            "detected_classes": r.detected_classes,
            "before_image": r.before_image,
            "after_image": r.after_image,
            "overlay_path": r.overlay_path,
            "binary_mask_path": r.binary_mask_path,
            "before_semantic_path": r.before_semantic_path,
            "after_semantic_path": r.after_semantic_path,
            "chart_path": r.chart_path,
            "json_path": r.json_path,
            "pdf_path": r.pdf_path,
            "created_at": str(r.created_at),
        }
        for r in records
    ]


@router.delete("/history/{analysis_id}")
def delete_history(analysis_id: int, db: Session = Depends(get_db)):
    record = (
        db.query(AnalysisHistory)
        .filter(AnalysisHistory.id == analysis_id)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis record not found")

    image_name = record.image_name or ""
    if image_name:
        for folder in (OUTPUT_FOLDER, UPLOAD_FOLDER):
            for path in folder.glob(f"{image_name}*"):
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Could not delete file: %s", path)

    db.delete(record)
    db.commit()
    return {"success": True, "deleted_id": analysis_id}


@router.get("/download/{image_name}")
def download_outputs(image_name: str):
    if not SAFE_IMAGE_NAME.match(image_name):
        raise HTTPException(status_code=400, detail="Invalid image name")
    files = sorted(OUTPUT_FOLDER.glob(f"{image_name}*"))
    uploads = sorted(UPLOAD_FOLDER.glob(f"{image_name}*"))
    if not files and not uploads:
        raise HTTPException(status_code=404, detail="No output files found")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(str(file_path), arcname=file_path.name)
        for file_path in uploads:
            zf.write(str(file_path), arcname=f"uploads/{file_path.name}")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{image_name}_results.zip"'
        },
    )


@router.get("/progress/{job_id}")
def get_progress(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/predict")
async def predict(
    request: Request,
    before: UploadFile = File(...),
    after: UploadFile = File(...),
):
    """Start an uploaded-image analysis in the background and return a job id.

    The live progress is polled via GET /predict-result/{job_id}, which
    returns the completed result (and writes the history row) when done.
    """
    image_id = f"img_{str(uuid.uuid4())[:8]}"
    before_path = UPLOAD_FOLDER / f"{image_id}_before.png"
    after_path = UPLOAD_FOLDER / f"{image_id}_after.png"
    job_id = image_id

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI pipeline not loaded",
        )

    try:
        content_before = await before.read()
        content_after = await after.read()
        before_path.write_bytes(content_before)
        after_path.write_bytes(content_after)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded files: {e}")

    create_job(job_id, "upload", {"image_name": image_id})
    update_job(job_id, 5, "Images uploaded")

    def _run_sync():
        return pipeline.run(
            before_image_path=str(before_path),
            after_image_path=str(after_path),
            output_dir=str(OUTPUT_FOLDER),
            image_name=image_id,
            job_id=job_id,
        )

    submit_job(job_id, _run_sync)

    return {"success": True, "job_id": job_id, "image_name": image_id, "status": "running"}


@router.get("/predict-result/{job_id}")
def predict_result(job_id: str, db: Session = Depends(get_db)):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "running":
        return {"job_id": job_id, "status": "running", "progress": job["progress"], "stage": job["stage"]}
    if job["status"] == "queued":
        return {"job_id": job_id, "status": "queued", "progress": job["progress"], "stage": job["stage"]}
    if job["status"] == "interrupted":
        return {
            "job_id": job_id,
            "status": "error",
            "detail": "Job was interrupted by a server restart; please run it again.",
        }
    if job["status"] == "error":
        detail = job.get("error") or job.get("stage") or "Analysis failed"
        return {"job_id": job_id, "status": "error", "detail": detail}

    result = store.get_result(job_id)
    if result is None:
        if job["status"] == "completed":
            return {
                "job_id": job_id,
                "status": "error",
                "detail": "The job finished but its result is unavailable; please run the analysis again.",
            }
        return {"job_id": job_id, "status": "running", "progress": job["progress"], "stage": job["stage"]}

    if result.get("history_id") is None:
        existing = (
            db.query(AnalysisHistory)
            .filter(AnalysisHistory.image_name == job_id)
            .first()
        )
        if existing is not None:
            result["history_id"] = existing.id
            result["report_name"] = existing.report_name
            store.update_result(job_id, result)
        else:
            objects = result.get("objects", [])
            detected_classes = sorted({obj.get("class_name", "Unknown") for obj in objects})
            count = db.query(AnalysisHistory).count()
            report_name = f"Image Report {count + 1}"

            record = AnalysisHistory(
                report_name=report_name,
                image_name=job_id,
                source="upload",
                changed_pixels=result["changed_pixels"],
                change_percentage=result["change_percentage"],
                processing_time=result["processing_time"],
                object_count=len(objects),
                detected_classes=",".join(detected_classes),
                before_image=f"/static/uploads/{job_id}_before.png",
                after_image=f"/static/uploads/{job_id}_after.png",
                overlay_path=f"/static/outputs/{job_id}_overlay.png",
                binary_mask_path=f"/static/outputs/{job_id}_binary_mask.png",
                before_semantic_path=f"/static/outputs/{job_id}_before_semantic.png",
                after_semantic_path=f"/static/outputs/{job_id}_after_semantic.png",
                chart_path=f"/static/outputs/{job_id}_chart.png",
                json_path=f"/static/outputs/{job_id}.json",
                pdf_path=f"/static/outputs/{job_id}.pdf",
                created_at=datetime.now(),
            )
            db.add(record)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                record = (
                    db.query(AnalysisHistory)
                    .filter(AnalysisHistory.image_name == job_id)
                    .first()
                )
                if record is None:
                    raise
            db.refresh(record)
            result["history_id"] = record.id
            result["report_name"] = report_name
            store.update_result(job_id, result)

    image_id = job_id
    return {
        "success": True,
        "report_name": result.get("report_name", f"Image Report"),
        "image_name": image_id,
        "source": "upload",
        "image": result["image_size"],
        "processing_time": result["processing_time"],
        "change_detection": {
            "changed_pixels": result["changed_pixels"],
            "change_percentage": result["change_percentage"],
        },
        "objects_detected": len(result.get("objects", [])),
        "detected_classes": sorted({obj.get("class_name", "Unknown") for obj in result.get("objects", [])}),
        "objects": result["objects"],
        "report": result["report"],
        "files": {
            "overlay": f"/static/outputs/{image_id}_overlay.png",
            "json": f"/static/outputs/{image_id}.json",
            "pdf": f"/static/outputs/{image_id}.pdf",
            "chart": f"/static/outputs/{image_id}_chart.png",
            "binary_mask": f"/static/outputs/{image_id}_binary_mask.png",
            "before_semantic": f"/static/outputs/{image_id}_before_semantic.png",
            "after_semantic": f"/static/outputs/{image_id}_after_semantic.png",
            "confidence_map": f"/static/outputs/{image_id}_confidence.png",
            "severity_map": f"/static/outputs/{image_id}_severity.png",
        },
        "severity": result.get("severity", {}),
        "job_id": job_id,
        "history_id": result["history_id"],
        "status": "completed",
    }


@router.post("/map-predict")
async def map_predict(request: Request):
    """Start a map-based analysis in the background and return the job id.

    The result is fetched later via GET /map-result/{job_id}.
    """
    body = await request.json()
    try:
        lat = float(body.get("lat"))
        lng = float(body.get("lng"))
        buffer_deg = float(body.get("buffer_deg", 0.01))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="lat, lng and buffer_deg must be numbers")
    date1 = str(body.get("date1", "2024-01-01"))
    date2 = str(body.get("date2", "2024-06-01"))

    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lng <= 180.0):
        raise HTTPException(status_code=400, detail="lat/lng out of range")
    if not (0.0 < buffer_deg <= 5.0):
        raise HTTPException(status_code=400, detail="buffer_deg must be between 0 and 5")
    for label, value in (("date1", date1), ("date2", date2)):
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail=f"{label} must be formatted YYYY-MM-DD")
    if date1 >= date2:
        raise HTTPException(status_code=400, detail="date1 must be before date2")

    job_id = f"map_{str(uuid.uuid4())[:8]}"
    create_job(job_id, "map", {"lat": lat, "lng": lng, "buffer_deg": buffer_deg})
    update_job(job_id, 5, "Fetching Sentinel-2 imagery")

    def _run_sync():
        from src.satellite_fetcher import fetch_satellite_pair

        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            raise RuntimeError("AI pipeline not loaded")

        update_job(job_id, 10, "Searching Sentinel-2 archive")
        before_img, after_img, meta, spectral_change_mask = fetch_satellite_pair(
            lat, lng, buffer_deg, date1, date2
        )
        if before_img is None or after_img is None:
            raise RuntimeError("Could not download satellite imagery for the selected region")

        before_path = UPLOAD_FOLDER / f"{job_id}_before.png"
        after_path = UPLOAD_FOLDER / f"{job_id}_after.png"
        cv2.imwrite(str(before_path), before_img)
        cv2.imwrite(str(after_path), after_img)

        result = pipeline.run(
            before_image_path=str(before_path),
            after_image_path=str(after_path),
            output_dir=str(OUTPUT_FOLDER),
            image_name=job_id,
            additional_change_mask=spectral_change_mask,
            job_id=job_id,
            bounds=meta.get("bounds"),
        )
        result["satellite_meta"] = meta
        result["files"]["before_image"] = f"/static/uploads/{job_id}_before.png"
        result["files"]["after_image"] = f"/static/uploads/{job_id}_after.png"
        result["history_id"] = None
        return result

    submit_job(job_id, _run_sync)

    return {"success": True, "job_id": job_id, "status": "running"}


@router.get("/map-result/{job_id}")
def map_result(job_id: str, db: Session = Depends(get_db)):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "running":
        return {"job_id": job_id, "status": "running", "progress": job["progress"], "stage": job["stage"]}
    if job["status"] == "queued":
        return {"job_id": job_id, "status": "queued", "progress": job["progress"], "stage": job["stage"]}
    if job["status"] == "interrupted":
        return {
            "job_id": job_id,
            "status": "error",
            "detail": "Job was interrupted by a server restart; please run it again.",
        }
    if job["status"] == "error":
        detail = job.get("error") or job.get("stage") or "Analysis failed"
        return {"job_id": job_id, "status": "error", "detail": detail}

    result = store.get_result(job_id)
    if result is None:
        if job["status"] == "completed":
            return {
                "job_id": job_id,
                "status": "error",
                "detail": "The job finished but its result is unavailable; please run the analysis again.",
            }
        return {"job_id": job_id, "status": "running", "progress": job["progress"], "stage": job["stage"]}

    if result.get("history_id") is None:
        existing = (
            db.query(AnalysisHistory)
            .filter(AnalysisHistory.image_name == job_id)
            .first()
        )
        if existing is not None:
            result["history_id"] = existing.id
            result["report_name"] = existing.report_name
            store.update_result(job_id, result)
        else:
            objects = result.get("objects", [])
            detected_classes = sorted({obj.get("class_name", "Unknown") for obj in objects})
            count = db.query(AnalysisHistory).count()
            report_name = f"Map Report {count + 1}"

            record = AnalysisHistory(
                report_name=report_name,
                image_name=job_id,
                source="map",
                changed_pixels=result["changed_pixels"],
                change_percentage=result["change_percentage"],
                processing_time=result["processing_time"],
                object_count=len(objects),
                detected_classes=",".join(detected_classes),
                before_image=f"/static/uploads/{job_id}_before.png",
                after_image=f"/static/uploads/{job_id}_after.png",
                overlay_path=f"/static/outputs/{job_id}_overlay.png",
                binary_mask_path=f"/static/outputs/{job_id}_binary_mask.png",
                before_semantic_path=f"/static/outputs/{job_id}_before_semantic.png",
                after_semantic_path=f"/static/outputs/{job_id}_after_semantic.png",
                chart_path=f"/static/outputs/{job_id}_chart.png",
                json_path=f"/static/outputs/{job_id}.json",
                pdf_path=f"/static/outputs/{job_id}.pdf",
                created_at=datetime.now(),
            )
            db.add(record)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                record = (
                    db.query(AnalysisHistory)
                    .filter(AnalysisHistory.image_name == job_id)
                    .first()
                )
                if record is None:
                    raise
            db.refresh(record)
            result["history_id"] = record.id
            result["report_name"] = report_name
            store.update_result(job_id, result)

    return {
        "success": True,
        "report_name": result.get("report_name", f"Map Report"),
        "image_name": job_id,
        "source": "map",
        "satellite_meta": result.get("satellite_meta"),
        "image": result["image_size"],
        "processing_time": result["processing_time"],
        "change_detection": {
            "changed_pixels": result["changed_pixels"],
            "change_percentage": result["change_percentage"],
        },
        "objects_detected": len(result.get("objects", [])),
        "detected_classes": sorted({obj.get("class_name", "Unknown") for obj in result.get("objects", [])}),
        "objects": result["objects"],
        "report": result["report"],
        "files": result["files"],
        "job_id": job_id,
        "history_id": result["history_id"],
        "status": "completed",
    }
