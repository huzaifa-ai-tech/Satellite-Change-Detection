import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from src.config import Config

logger = logging.getLogger("satellite.segmenter")


class LoveDASegmenter:
    TILE_SIZE = 512
    TILE_OVERLAP = 128
    MIN_TILE = 64

    def __init__(self, checkpoint_path=None, model_variant="b2"):
        if checkpoint_path is None:
            checkpoint_path = Config.SEGFORMER_CHECKPOINT

        self.model_variant = model_variant.lower()
        logger.info("Loading SegFormer %s on %s", self.model_variant.upper(), Config.DEVICE)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SegFormer checkpoint missing: {ckpt_path}")

        self.device = Config.DEVICE
        self.checkpoint_path = checkpoint_path

        try:
            self.processor = SegformerImageProcessor.from_pretrained(
                checkpoint_path, local_files_only=True
            )
        except Exception:
            logger.warning("Using default SegFormer processor")
            self.processor = SegformerImageProcessor(
                do_resize=True, size={"height": 512, "width": 512}, do_normalize=True
            )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint_path, local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info("SegFormer %s ready", self.model_variant.upper())

    def _run_model(self, rgb_array):
        image = Image.fromarray(rgb_array)
        width, height = image.size

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )

        probabilities = torch.softmax(logits, dim=1)
        confidence, semantic = torch.max(probabilities, dim=1)

        semantic_mask = semantic[0].cpu().numpy().astype(np.uint8)
        confidence_map = confidence[0].float().cpu().numpy()

        return semantic_mask, confidence_map

    @staticmethod
    def _remap(semantic_mask):
        return np.where(semantic_mask <= 1, 0, semantic_mask - 1).astype(np.uint8)

    @staticmethod
    def _tile_ranges(size, tile, step):
        ranges = []
        pos = 0
        while pos < size:
            end = min(pos + tile, size)
            if end - pos < LoveDASegmenter.MIN_TILE and pos > 0:
                pos = max(0, end - tile)
                end = min(pos + tile, size)
            ranges.append((pos, end))
            pos += step
            if pos >= size:
                break
        return ranges

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        rgb_image = np.array(image)
        h, w = rgb_image.shape[:2]

        if h <= self.TILE_SIZE and w <= self.TILE_SIZE:
            semantic_mask, confidence_map = self._run_model(rgb_image)
            semantic_mask = self._remap(semantic_mask)
            return rgb_image, semantic_mask, confidence_map

        step = self.TILE_SIZE - self.TILE_OVERLAP
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        for y0, y1 in self._tile_ranges(h, self.TILE_SIZE, step):
            for x0, x1 in self._tile_ranges(w, self.TILE_SIZE, step):
                patch = rgb_image[y0:y1, x0:x1]
                psem, pconf = self._run_model(patch)
                psem = self._remap(psem)

                region = (slice(y0, y1), slice(x0, x1))
                take = pconf >= confidence_map[region]
                semantic_mask[region] = np.where(take, psem, semantic_mask[region])
                confidence_map[region] = np.maximum(confidence_map[region], pconf)

        return rgb_image, semantic_mask, confidence_map
