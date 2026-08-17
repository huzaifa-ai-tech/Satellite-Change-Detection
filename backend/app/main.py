import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import engine, Base
from app.jobs import store
from app.routes import router
from src.pipeline import SatellitePipeline
from src.satellite_classes import SATELLITE_CLASSES
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satellite")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)

Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Satellite AI Backend")
    store.mark_interrupted()
    Config.check_paths()
    try:
        logger.info("Loading AI Pipeline...")
        pipeline = SatellitePipeline(SATELLITE_CLASSES)
        app.state.pipeline = pipeline
        logger.info("Pipeline loaded: %s", Config.summary())
    except Exception as e:
        logger.error("Pipeline loading failed: %s", e)
        app.state.pipeline = None
    yield
    logger.info("Satellite AI Backend Shutdown")


app = FastAPI(
    title="Satellite Change Detection API",
    version="1.0.0",
    description="AI satellite image analysis with ready-to-use change detection",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    pipeline_status = "loaded" if getattr(app.state, "pipeline", None) else "not loaded"
    return {
        "message": "Satellite Change Detection API",
        "version": "1.0.0",
        "status": "running",
        "database": "SQLite",
        "pipeline": pipeline_status,
        "config": Config.summary(),
    }
