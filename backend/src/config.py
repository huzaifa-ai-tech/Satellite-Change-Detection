import logging
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("satellite.config")

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # Device override: DEVICE=cpu, DEVICE=cuda, DEVICE=cuda:1 or DEVICE=0
    # (a bare integer means the CUDA device index). When unset, auto-select
    # CUDA if available, else CPU.
    _DEVICE_ENV = os.getenv("DEVICE", "").strip()
    if _DEVICE_ENV:
        _device_spec = _DEVICE_ENV
        if _DEVICE_ENV.isdigit():
            _device_spec = f"cuda:{_DEVICE_ENV}"
        elif _DEVICE_ENV.startswith("cuda") and ":" not in _DEVICE_ENV:
            _device_spec = "cuda"
        DEVICE = torch.device(_device_spec)
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = os.getenv("CHANGEFORMER_MODEL", "ChangeFormerV6")
    EMBED_DIM = int(os.getenv("EMBED_DIM", "256"))

    CHANGEFORMER_CHECKPOINT = os.getenv(
        "CHANGEFORMER_CHECKPOINT",
        str(BASE_DIR / "ChangeFormer" / "checkpoints" / "ChangeFormer_LEVIR" / "best_ckpt.pt"),
    )

    PATCH_SIZE = int(os.getenv("PATCH_SIZE", "384"))
    # Stride smaller than the patch size gives tile overlap, which removes
    # seams and missed changes at tile boundaries.
    PATCH_STRIDE = int(os.getenv("PATCH_STRIDE", "300"))
    OBJECT_CHANGE_THRESHOLD = float(os.getenv("OBJECT_CHANGE_THRESHOLD", "0.30"))
    ONLY_CHANGED_OBJECTS = os.getenv("ONLY_CHANGED_OBJECTS", "true").lower() == "true"
    # YOLO26-OBB fires "large-vehicle" not only on real trucks but also on
    # boxes that merge two adjacent cars into one detection. A real truck is
    # a single solid dark mass: require the largest dark blob inside the box
    # to cover at least this fraction of the box area.
    TRUCK_MIN_SOLID_FRACTION = float(os.getenv("TRUCK_MIN_SOLID_FRACTION", "0.35"))
    # Minimum size of a coherent semantic class-transition region for it to
    # count as a real land-cover change (forest -> barren, field ->
    # building, ...). Smaller scattered flips on AI-generated pairs are
    # texture noise, not real changes.
    TRANSITION_MIN_AREA = int(os.getenv("TRANSITION_MIN_AREA", "2500"))
    # For transition regions smaller than TRANSITION_MIN_AREA, minimum mean
    # pixel difference (after illumination normalization) required. Real
    # changes move actual image content; texture noise does not.
    TRANSITION_MIN_DIFF = float(os.getenv("TRANSITION_MIN_DIFF", "10"))

    SEGFORMER_CHECKPOINT = os.getenv("SEGFORMER_CHECKPOINT", str(BASE_DIR / "loveda_segformer_b2"))
    SEGFORMER_VARIANT = os.getenv("SEGFORMER_VARIANT", "b2")

    UPLOAD_FOLDER = str(BASE_DIR / "app" / "static" / "uploads")
    OUTPUT_FOLDER = str(BASE_DIR / "app" / "static" / "outputs")

    MODEL_TIER = os.getenv("MODEL_TIER", "ready")

    @staticmethod
    def check_paths():
        for path in [Config.CHANGEFORMER_CHECKPOINT, Config.SEGFORMER_CHECKPOINT]:
            if not os.path.exists(path):
                logger.warning("Missing path: %s", path)

    @staticmethod
    def summary():
        return {
            "device": str(Config.DEVICE),
            "tier": Config.MODEL_TIER,
            "change_detector": Config.MODEL_NAME,
            "segmenter": f"SegFormer-{Config.SEGFORMER_VARIANT.upper()}",
            "object_detector": "YOLO26-OBB (DOTA)",
            "pipeline": "change-first (ChangeFormerV6 mask -> SegFormer land-cover -> YOLO vehicles)",
            "only_changed_objects": Config.ONLY_CHANGED_OBJECTS,
        }
