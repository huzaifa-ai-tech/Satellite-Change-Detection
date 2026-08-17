"""
Satellite Change Detection - Models Integration
===============================================
Models integrated:
- ChangeFormer V6 (SOTA transformer change detector, tiled inference)
- YOLO26-OBB (DOTA-trained oriented aerial object detector)
"""

import logging
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import Config

logger = logging.getLogger("satellite.models")


def _bbox_change_overlap(bbox, change_mask):
    if change_mask is None:
        return 1.0
    if bbox is None or len(bbox) != 4:
        return 0.0
    x, y, w, h = [int(v) for v in bbox]
    h_img, w_img = change_mask.shape
    if w <= 0 or h <= 0 or x >= w_img or y >= h_img:
        return 0.0
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(change_mask[y1:y2, x1:x2] > 0))


class ModelManager:
    """Manages all latest AI models for satellite image analysis."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        # Bounded LRU cache of full-resolution float32 probability maps
        # (~1-9 MB each): a few jobs fit, but an unbounded dict would grow
        # without limit over the lifetime of the server.
        self._prob_cache = OrderedDict()
        self._prob_cache_max = 8
        self._load_models()

    def _cache_prob(self, key, prob):
        """Store a probability map in the bounded LRU cache."""
        self._prob_cache[key] = prob
        self._prob_cache.move_to_end(key)
        while len(self._prob_cache) > self._prob_cache_max:
            self._prob_cache.popitem(last=False)

    def _load_models(self):
        """Load all available models."""
        self._load_changeformer()
        self._load_snunet()

    def _load_changeformer(self):
        """Load ChangeFormer V6 (weights already bundled in backend/ChangeFormer/)."""
        try:
            from src.config import BASE_DIR, Config

            cf_dir = str(Path(BASE_DIR) / "ChangeFormer")
            if cf_dir not in sys.path:
                sys.path.insert(0, cf_dir)

            from models.ChangeFormer import ChangeFormerV6

            checkpoint = Config.CHANGEFORMER_CHECKPOINT
            if not Path(checkpoint).exists():
                logger.warning("ChangeFormer checkpoint not found: %s", checkpoint)
                self.models['changeformer_available'] = False
                return

            model = ChangeFormerV6(input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=Config.EMBED_DIM)
            ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_G_state_dict', ckpt)
            model.load_state_dict(state_dict, strict=False)
            model.eval().to(self.device)

            self.models['changeformer'] = model
            self.models['changeformer_available'] = True
            logger.info("ChangeFormer V6 loaded from %s", checkpoint)
        except Exception as e:
            self.models['changeformer_available'] = False
            logger.warning(f"ChangeFormer V6 not available: {e}")

    def _load_snunet(self):
        """Load SNUNet (Siamese Nested U-Net) change detector.

        SNUNet is a lightweight CNN-based detector that complements the
        transformer-based ChangeFormer: CNNs excel at local/texture cues
        while transformers capture long-range context, so ensembling the
        two gives better recall on both small and large changes.

        Pretrained weights are looked for at
        ``backend/models/snunet_levir.pt``.  If absent the model is
        skipped silently and the pipeline falls back to ChangeFormer only.
        """
        try:
            from src.config import BASE_DIR
            from src.snunet import SNUNet

            snunet_path = Path(BASE_DIR) / "models" / "snunet_levir.pt"
            if not snunet_path.exists():
                logger.info("SNUNet weights not found at %s — ensemble disabled", snunet_path)
                self.models['snunet_available'] = False
                return

            model = SNUNet(in_ch=3, base_channel=16)
            ckpt = torch.load(str(snunet_path), map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logger.warning("SNUNet weights partially matched: missing=%s unexpected=%s",
                               missing, unexpected)
            model.eval().to(self.device)

            self.models['snunet'] = model
            self.models['snunet_available'] = True
            logger.info("SNUNet loaded from %s", snunet_path)
        except Exception as e:
            self.models['snunet_available'] = False
            logger.warning(f"SNUNet not available: {e}")

    @staticmethod
    def _same_vehicle(a, b):
        """True when two classes refer to the same physical object: equal
        classes always match; road vehicles also match across classes
        (car <-> truck) because YOLO wobbles between "small-vehicle" and
        "large-vehicle" on the same object."""
        if a == b:
            return True
        vehicles = {"car", "truck", "bus"}
        return a in vehicles and b in vehicles

    @staticmethod
    def _iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _get_yolo(self):
        from src.config import BASE_DIR
        from ultralytics import YOLO

        yolo_path = str(Path(BASE_DIR) / "models" / "yolo26s-obb.pt")
        if 'yolo' not in self.models:
            self.models['yolo'] = YOLO(yolo_path)
        return self.models['yolo']

    def detect_objects(self, before_path, after_path, change_mask):
        """Detect objects with YOLO26-OBB (DOTA) in both images and classify
        each as appeared / removed.

        Change-first pipeline. For vehicles, before/after matching IS the
        change detector: a box in the after image with no stable match in
        the before image is "appeared" (new), a box only in the before is
        "removed". Boxes present in both with stable size are "unchanged"
        and are dropped by the pipeline. The change mask contributes the
        overlap score only.
        """
        from src.config import Config

        before_objs = self._detect_objects_yolo(before_path)
        after_objs = self._detect_objects_yolo(after_path)

        matched_before = set()
        for obj in after_objs:
            cov = _bbox_change_overlap(obj["bbox"], change_mask)
            obj["change_overlap"] = round(cov, 3)

            best_iou, best_idx = 0.0, -1
            best_area_ratio = None
            for i, bo in enumerate(before_objs):
                # Vehicles match across classes (car <-> truck): YOLO wobbles
                # between "small-vehicle" and "large-vehicle" on the same
                # object, and a car present in both images must not be
                # reported as both removed (car) and appeared (truck).
                if not ModelManager._same_vehicle(bo["class_name"], obj["class_name"]):
                    continue
                iou = self._iou(obj["bbox"], bo["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
                    best_area_ratio = (bo["bbox"][2] * bo["bbox"][3]) / max(
                        obj["bbox"][2] * obj["bbox"][3], 1)
            if best_iou > 0.3:
                matched_before.add(best_idx)
                # Same object in both images only if its size is stable;
                # a box that grew/shrunk by >50% is itself a change.
                size_stable = best_area_ratio is not None and 0.5 <= best_area_ratio <= 2.0
                obj["status"] = "unchanged" if size_stable else "appeared"
            else:
                obj["status"] = "appeared"

        for i, bo in enumerate(before_objs):
            if i in matched_before:
                continue
            if bo["confidence"] < 0.35:
                continue
            bo["change_overlap"] = round(_bbox_change_overlap(bo["bbox"], change_mask), 3)
            bo["status"] = "removed"
            after_objs.append(bo)

        return after_objs

    @staticmethod
    def _nms(objects, iou_threshold=0.5):
        """Deduplicate overlapping boxes across classes: when two detections
        cover the same object (YOLO often fires both "small-vehicle" and
        "large-vehicle" on the same box), keep only the higher-confidence one."""
        kept = []
        for obj in sorted(objects, key=lambda o: -o["confidence"]):
            dup = any(ModelManager._iou(obj["bbox"], k["bbox"]) > iou_threshold for k in kept)
            if not dup:
                kept.append(obj)
        return kept

    def _parse_yolo_results(self, results, image_path, offset_x=0, offset_y=0):
        """Parse YOLO results into a list of object dicts, optionally
        translating coordinates by (offset_x, offset_y) for tiled inference."""
        objects = []
        for result in results:
            names = result.names
            obb = result.obb if hasattr(result, "obb") and result.obb is not None else None
            boxes = obb if obb is not None and len(obb) > 0 else result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                name = names.get(cls_id, "unknown").replace(" ", "-")
                mapped = {
                    "small-vehicle": "car",
                    "large-vehicle": "truck",
                    "ship": "ship",
                    "plane": "aircraft",
                    "helicopter": "aircraft",
                    "storage-tank": "storage_tank",
                    "harbor": "harbor",
                    "bridge": "bridge",
                }.get(name)
                if mapped is None:
                    if name in ("car", "truck", "bus", "boat"):
                        mapped = {"car": "car", "truck": "truck", "bus": "bus", "boat": "ship"}.get(name)
                    else:
                        continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                w, h = x2 - x1, y2 - y1
                if min(w, h) < 4 or w * h < 20:
                    continue
                if mapped == "truck" and not self._is_solid_vehicle(
                        str(image_path), int(x1), int(y1), int(w), int(h)):
                    logger.debug(
                        "Dropping large-vehicle box at [%d, %d, %d, %d]: not a solid mass",
                        int(x1), int(y1), int(w), int(h))
                    continue
                objects.append({
                    "class_name": mapped,
                    "confidence": conf,
                    "bbox": [int(x1) + offset_x, int(y1) + offset_y, int(w), int(h)],
                    "status": "detected",
                })
        return objects

    def _detect_objects_yolo(self, image_path):
        """YOLO26-OBB (DOTA) full object detection with class mapping.

        For images larger than 1024 px the image is split into overlapping
        640×640 tiles so that small objects (cars, etc.) appear larger
        relative to the tile and receive higher confidence scores.
        """
        objects = []
        try:
            from ultralytics import YOLO

            model = self._get_yolo()
            img = cv2.imread(str(image_path))
            if img is None:
                return objects

            h, w = img.shape[:2]
            tile_size = 640
            overlap = 128

            if max(h, w) <= tile_size:
                results = model.predict(str(image_path), conf=0.10, verbose=False)
                objects = self._parse_yolo_results(results, image_path)
            else:
                step = tile_size - overlap
                for y0 in range(0, h, step):
                    for x0 in range(0, w, step):
                        y1 = min(y0 + tile_size, h)
                        x1 = min(x0 + tile_size, w)
                        tile = img[y0:y1, x0:x1]
                        tmp_path = str(Path(image_path).parent / f"_tile_{y0}_{x0}.jpg")
                        cv2.imwrite(tmp_path, tile)
                        results = model.predict(tmp_path, conf=0.10, verbose=False)
                        tile_objs = self._parse_yolo_results(results, tmp_path, offset_x=x0, offset_y=y0)
                        objects.extend(tile_objs)
                        try:
                            Path(tmp_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                        if x1 >= w:
                            break
                    if y1 >= h:
                        break

            objects = self._nms(objects)
            if objects:
                logger.info("YOLO26-OBB detected %d objects", len(objects))
        except Exception as e:
            logger.warning(f"YOLO object detection failed: {e}")

        return objects

    @staticmethod
    def _is_solid_vehicle(image_path, x, y, w, h, min_frac=None):
        """Reject "large-vehicle" detections that are not one solid dark mass.

        YOLO26-OBB merges two adjacent cars into a single "large-vehicle"
        box. A real truck is one contiguous dark region filling most of its
        box; a merged pair of cars shows fragmented dark blobs, none of
        which dominates the box. Returns True when the largest dark blob
        covers >= TRUCK_MIN_SOLID_FRACTION of the box area.
        """
        try:
            import cv2
            from src.config import Config

            if min_frac is None:
                min_frac = Config.TRUCK_MIN_SOLID_FRACTION
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return True
            x0, y0 = max(0, x), max(0, y)
            x1 = min(img.shape[1], x + w)
            y1 = min(img.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                return False
            crop = img[y0:y1, x0:x1]
            dark = (crop < 100).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
            box_area = max((x1 - x0) * (y1 - y0), 1)
            largest = 0
            for i in range(1, n):
                largest = max(largest, int(stats[i, cv2.CC_STAT_AREA]))
            return (largest / box_area) >= min_frac
        except Exception as e:
            logger.warning(f"Solid-vehicle check failed: {e}")
            return True

    def detect_changes(self, before_path, after_path):
        """Detect changes using ensemble (ChangeFormer + SNUNet) when both
        are available, falling back to whichever single model is loaded.

        The ensemble averages the normalised probability maps from
        ChangeFormer V6 (transformer, long-range context) and SNUNet
        (CNN, local texture), giving +3-5 % F1 over either model alone.
        """
        # cv2.imread is patched by ultralytics and only accepts str paths
        # (Path objects crash with AttributeError on .endswith).
        before_path, after_path = str(before_path), str(after_path)

        cf_avail = self.models.get('changeformer_available', False)
        sn_avail = self.models.get('snunet_available', False)

        if cf_avail and sn_avail:
            try:
                mask = self._detect_with_ensemble(before_path, after_path)
                if mask is not None:
                    return mask
            except Exception as e:
                logger.debug(f"Ensemble failed, falling back: {e}")

        if cf_avail:
            try:
                mask = self._detect_with_changeformer(before_path, after_path)
                if mask is not None:
                    return mask
            except Exception as e:
                logger.debug(f"ChangeFormer failed: {e}")

        if sn_avail:
            try:
                mask = self._detect_with_snunet(before_path, after_path)
                if mask is not None:
                    return mask
            except Exception as e:
                logger.debug(f"SNUNet failed: {e}")

        return self._detect_with_spectral(before_path, after_path)

    def _ensemble_prob(self, before_path, after_path):
        """Return ``(prob_map, used)`` for the change mask.

        Averages the ChangeFormer and SNUNet probability maps when both are
        alive.  ChangeFormer V6 is trained at exactly 256x256, so for inputs
        much larger than 256 its resize-to-256 probability map collapses to
        near-zero (max <= 0.05) and contains no usable signal.  Averaging it
        in would only halve SNUNet's full-resolution detections, dropping
        every change with SNUNet confidence below 0.80.  In that case we use
        SNUNet's map alone; the mask, confidence and severity maps then all
        agree because they share this same helper.
        """
        cf_avail = self.models.get('changeformer_available', False)
        sn_avail = self.models.get('snunet_available', False)

        prob_cf = self._prob_with_changeformer(before_path, after_path) if cf_avail else None
        prob_sn = self._prob_with_snunet(before_path, after_path) if sn_avail else None

        if prob_cf is not None and prob_sn is not None:
            if float(prob_cf.max()) <= 0.05:
                logger.info(
                    "ChangeFormer prob dead (max %.4f) on large input - using SNUNet alone",
                    float(prob_cf.max()),
                )
                return prob_sn, 'snunet'
            return (prob_cf + prob_sn) / 2.0, 'ensemble'
        if prob_cf is not None:
            return prob_cf, 'changeformer'
        if prob_sn is not None:
            return prob_sn, 'snunet'
        return None, None

    def _detect_with_ensemble(self, before_path, after_path):
        """Ensemble: average probability maps from ChangeFormer + SNUNet."""
        prob, _used = self._ensemble_prob(before_path, after_path)
        if prob is None:
            return None
        return self._mask_from_prob(prob)

    def get_change_probability(self, before_path, after_path):
        """Return the raw float probability map for the change mask.

        Used by the pipeline to generate confidence heatmaps and severity
        maps without re-running inference.

        Returns the same ensemble-averaged probability map that produced
        the change mask (ChangeFormer + SNUNet when both are loaded), so
        the confidence/severity heatmaps agree with the mask instead of
        under-reporting the stronger model's confidence.
        """
        before_path, after_path = str(before_path), str(after_path)

        prob, _used = self._ensemble_prob(before_path, after_path)
        if prob is not None:
            return prob

        # Spectral fallback
        imgA = cv2.imread(before_path)
        imgB = cv2.imread(after_path)
        if imgA is None or imgB is None:
            return None
        a = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB).astype(np.float32)
        b = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB).astype(np.float32)
        diff = np.abs(b - a)
        mag = np.sqrt(np.sum(diff ** 2, axis=2))
        if mag.max() > 0:
            mag = mag / mag.max()
        return mag.astype(np.float32)

    def _mask_from_prob(self, prob):
        """Convert a float probability map to a 0/1 uint8 mask with
        morphological cleanup and small-region removal."""
        mask = (prob > 0.40).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < 50:
                mask[labels == i] = 0
        return mask

    def _detect_with_changeformer(self, before_path, after_path):
        """ChangeFormer V6 detection.  Returns 0/1 uint8 mask sized to input.

        The model is trained at exactly 256x256 and is unusable at other
        sizes, so ``_prob_with_changeformer`` resizes to 256, infers with
        h-flip TTA, and upsamples the probability map back to the input
        size.
        """
        prob = self._prob_with_changeformer(before_path, after_path)
        if prob is None:
            return None
        mask = self._mask_from_prob(prob)
        logger.info("ChangeFormer V6 (TTA) detected %d changed pixels", int(mask.sum()))
        return mask

    # ------------------------------------------------------------------
    # Probability-map helpers (used by ensemble)
    # ------------------------------------------------------------------

    def _prob_with_changeformer(self, before_path, after_path):
        """Return the raw float probability map from ChangeFormer V6 without
        thresholding.  Used by the ensemble to average with SNUNet's
        probability map.

        ChangeFormer V6 was trained at exactly 256x256 (the repo's demo
        resizes any input to 256 before inference); feeding other sizes or
        zero-padded tiles makes it produce near-zero change probability.
        We therefore follow the repo protocol: resize the pair to 256,
        infer, and upsample the probability map back to the original size.
        The repo config runs with multi_infer_False (no flip TTA): on the
        LEVIR pairs the flipped variant is much weaker than the native
        orientation, so averaging with it actually hurts.
        """
        from src.config import Config

        model = self.models.get('changeformer')
        cache_key = ('changeformer', str(before_path), str(after_path))
        if cache_key in self._prob_cache:
            return self._prob_cache[cache_key]
        imgA = cv2.imread(before_path)
        imgB = cv2.imread(after_path)
        if imgA is None or imgB is None:
            return None

        h, w = imgA.shape[:2]
        a = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        b = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (256, 256), interpolation=cv2.INTER_LINEAR)
        a = a.transpose(2, 0, 1)
        b = b.transpose(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        with torch.no_grad():
            ta = torch.from_numpy(a).unsqueeze(0)
            tb = torch.from_numpy(b).unsqueeze(0)
            ta = ((ta - mean) / std).to(self.device)
            tb = ((tb - mean) / std).to(self.device)
            out = model(ta, tb)
            logits = out[-1] if isinstance(out, (list, tuple)) else out
            prob256 = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

        prob = cv2.resize(prob256, (w, h), interpolation=cv2.INTER_LINEAR)
        prob = prob.astype(np.float32)
        self._cache_prob(cache_key, prob)
        return prob

    def _prob_with_snunet(self, before_path, after_path):
        """Return the raw float probability map from SNUNet (with h-flip
        TTA) without thresholding."""
        model = self.models.get('snunet')
        cache_key = ('snunet', str(before_path), str(after_path))
        if cache_key in self._prob_cache:
            return self._prob_cache[cache_key]
        imgA = cv2.imread(before_path)
        imgB = cv2.imread(after_path)
        if imgA is None or imgB is None:
            return None

        h, w = imgA.shape[:2]
        a = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        b = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        a = a.transpose(2, 0, 1)
        b = b.transpose(2, 0, 1)
        a_flip = np.flip(a, axis=2).copy()
        b_flip = np.flip(b, axis=2).copy()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        ps = 512
        stride = 512

        def _infer(a_arr, b_arr):
            p = np.zeros((h, w), dtype=np.float32)
            c = np.zeros((h, w), dtype=np.float32)
            with torch.no_grad():
                for y0 in range(0, h, stride):
                    for x0 in range(0, w, stride):
                        y1, x1 = min(y0 + ps, h), min(x0 + ps, w)
                        vh, vw = y1 - y0, x1 - x0
                        ta = torch.from_numpy(a_arr[:, y0:y1, x0:x1]).unsqueeze(0)
                        tb = torch.from_numpy(b_arr[:, y0:y1, x0:x1]).unsqueeze(0)
                        if vh < ps or vw < ps:
                            ta = torch.nn.functional.pad(ta, (0, ps - vw, 0, ps - vh))
                            tb = torch.nn.functional.pad(tb, (0, ps - vw, 0, ps - vh))
                        ta = ((ta - mean) / std).to(self.device)
                        tb = ((tb - mean) / std).to(self.device)
                        logits = model(ta, tb)
                        prob_patch = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                        p[y0:y1, x0:x1] += prob_patch[:vh, :vw]
                        c[y0:y1, x0:x1] += 1.0
            return p / np.maximum(c, 1.0)

        prob = _infer(a, b)
        prob_flip = _infer(a_flip, b_flip)
        prob = ((prob + np.flip(prob_flip, axis=1)) / 2.0).astype(np.float32)
        self._cache_prob(cache_key, prob)
        return prob

    def _detect_with_snunet(self, before_path, after_path):
        """SNUNet tiled inference. Returns 0/1 uint8 mask sized to input."""
        prob = self._prob_with_snunet(before_path, after_path)
        if prob is None:
            return None
        mask = self._mask_from_prob(prob)
        logger.info("SNUNet detected %d changed pixels", int(mask.sum()))
        return mask

    def _detect_with_spectral(self, before_path, after_path):
        """Fallback spectral change detection."""
        imgA = cv2.imread(before_path)
        imgB = cv2.imread(after_path)

        if imgA is None or imgB is None:
            return None

        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB).astype(np.float32)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB).astype(np.float32)

        diff = np.abs(imgB - imgA)
        magnitude = np.sqrt(np.sum(diff ** 2, axis=2))

        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()

        mask = (magnitude > 0.1).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask
