import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.change_analyzer import ChangeAnalyzer
from src.chart_generator import create_class_chart, create_severity_chart
from src.config import Config
from src.logger import Logger
from src.loveda_visualizer import colorize
from src.object_detector import ObjectDetector, SpecialObjectDetector
from src.overlay import OverlayGenerator
from src.pdf_report import PDFReportGenerator
from src.report_generator import ReportGenerator
from src.satellite_classes import SATELLITE_CLASSES
from src.models import ModelManager, _bbox_change_overlap
from app.progress import create_job, update_job

logger = logging.getLogger("satellite.pipeline")

VEHICLE_IDS = {"car": 11, "ship": 12, "truck": 14, "bus": 15, "aircraft": 17,
               "storage_tank": 20, "harbor": 18, "bridge": 8, "container_crane": 19}


def _center_crop(image, height, width):
    """Center-crop `image` to (height, width). Returns the image unchanged
    when it is already at that size."""
    h, w = image.shape[:2]
    if h == height and w == width:
        return image
    y0 = (h - height) // 2
    x0 = (w - width) // 2
    return image[y0:y0 + height, x0:x0 + width]


def _apply_clahe(image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
    L channel of a BGR image in LAB space. Improves local contrast before
    histogram matching so structural differences are preserved."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _match_histogram(reference, target):
    """Histogram-match `target` to `reference` in LAB space so both images
    share the same illumination. Returns a copy of target."""
    if reference.shape != target.shape:
        target = cv2.resize(target, (reference.shape[1], reference.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    reference = _apply_clahe(reference)
    target = _apply_clahe(target)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    matched = tgt_lab.copy()
    for c in range(3):
        ref_hist, ref_edges = np.histogram(ref_lab[..., c], bins=256, range=(0, 256))
        tgt_hist, tgt_edges = np.histogram(tgt_lab[..., c], bins=256, range=(0, 256))
        ref_cdf = np.cumsum(ref_hist) / ref_hist.sum()
        tgt_cdf = np.cumsum(tgt_hist) / tgt_hist.sum()
        ref_bins = np.searchsorted(ref_cdf, tgt_cdf, side="left")
        lookup = ref_bins.clip(0, 255).astype(np.uint8)
        matched[..., c] = lookup[tgt_lab[..., c].astype(np.uint8)]
    return cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_LAB2BGR)


class SatellitePipeline:
    def __init__(self, class_names=None):
        self.logger = Logger()
        self.class_names = class_names if class_names else SATELLITE_CLASSES

        self.logger.info("Initializing Satellite Pipeline")

        self.logger.info("Loading AI models")
        self.model_manager = ModelManager()

        self.analyzer = ChangeAnalyzer(self.class_names)
        self.object_detector = ObjectDetector()
        self.special_detector = SpecialObjectDetector()
        self.overlay = OverlayGenerator()
        self.report_generator = ReportGenerator()
        self.pdf_generator = PDFReportGenerator()

        self.logger.info("Satellite Pipeline Ready")

    def run(self, before_image_path, after_image_path, output_dir=None, image_name="result",
            additional_change_mask=None, job_id=None, bounds=None):
        start_time = time.time()

        if output_dir is None:
            output_dir = Config.OUTPUT_FOLDER
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if job_id:
            create_job(job_id)

        before_image = cv2.imread(str(before_image_path))
        after_image = cv2.imread(str(after_image_path))

        if before_image is None:
            raise FileNotFoundError(f"Cannot read: {before_image_path}")
        if after_image is None:
            raise FileNotFoundError(f"Cannot read: {after_image_path}")
        if before_image.shape[0] != after_image.shape[0] or before_image.shape[1] != after_image.shape[1]:
            # Align differing dimensions by center-cropping the image(s) with
            # the larger size down to the common (smaller) size, so both
            # images cover the same area without distortion.
            target_h = min(before_image.shape[0], after_image.shape[0])
            target_w = min(before_image.shape[1], after_image.shape[1])
            self.logger.info("Cropping image(s) to common size (%d, %d)", target_h, target_w)
            before_image = _center_crop(before_image, target_h, target_w)
            after_image = _center_crop(after_image, target_h, target_w)
            # Persist the cropped before AND after images so model inference
            # reads the same dimensions for both sides — otherwise SNUNet's
            # tiled inference gets a cropped before vs an uncropped after and
            # the probability maps are spatially misaligned.
            cropped_before_path = output_dir / f"{image_name}_before_crop.png"
            cropped_after_path = output_dir / f"{image_name}_after_crop.png"
            cv2.imwrite(str(cropped_before_path), before_image)
            cv2.imwrite(str(cropped_after_path), after_image)
            before_image_path = cropped_before_path
            after_image_path = cropped_after_path

        height, width = before_image.shape[:2]
        raw_after_image = after_image.copy()

        # Normalize illumination: AI-generated or differently-lit before/after
        # pairs make every pixel look "changed" to both the change detector
        # and the segmenter. Histogram-match the after image to the before
        # image so the models compare true structure, not lighting.
        after_image = _match_histogram(before_image, after_image)
        norm_after_path = output_dir / f"{image_name}_after_norm.png"
        cv2.imwrite(str(norm_after_path), after_image)

        if job_id:
            update_job(job_id, 20, "Running change detection")
        self.logger.info("Running change detection")

        # Change detection runs on the *raw* after image.  Histogram
        # matching is kept only for the segmentation path below: matching
        # the after image to the before image erases real pixel evidence
        # (a newly built house is "renormalised away"), collapsing the
        # ChangeFormer/SNUNet probability maps to near-zero.  Pairs with
        # gross illumination differences (e.g. AI-generated) are barely
        # affected either way, so the raw pair is strictly safer.
        change_mask = self.model_manager.detect_changes(before_image_path, str(after_image_path))
        if change_mask is None:
            change_mask = np.zeros((height, width), dtype=np.uint8)

        if additional_change_mask is not None:
            if additional_change_mask.shape != change_mask.shape:
                additional_change_mask = cv2.resize(
                    additional_change_mask, (change_mask.shape[1], change_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            change_mask = np.maximum(change_mask, additional_change_mask).astype(np.uint8)

        changed_pixels = int(np.sum(change_mask > 0))
        total_pixels = height * width
        change_percentage = round((changed_pixels / total_pixels) * 100, 2)

        binary_mask = (change_mask * 255).astype(np.uint8)
        binary_path = output_dir / f"{image_name}_binary_mask.png"
        cv2.imwrite(str(binary_path), binary_mask)

        if bounds:
            try:
                from src.geo_export import write_geotiff

                rgb_before = cv2.cvtColor(before_image, cv2.COLOR_BGR2RGB)
                rgb_after = cv2.cvtColor(after_image, cv2.COLOR_BGR2RGB)
                write_geotiff(output_dir / f"{image_name}_before.tif", rgb_before, bounds)
                write_geotiff(output_dir / f"{image_name}_after.tif", rgb_after, bounds)
                write_geotiff(output_dir / f"{image_name}_change_mask.tif", binary_mask, bounds)
            except Exception as e:
                self.logger.warning("GeoTIFF export skipped: %s", e)

        # --- Confidence map & severity analysis ---
        prob_map = self.model_manager.get_change_probability(before_image_path, str(after_image_path))
        if prob_map is not None:
            if prob_map.shape != (height, width):
                prob_map = cv2.resize(prob_map, (width, height), interpolation=cv2.INTER_LINEAR)

            # Confidence heatmap (JET colormap)
            prob_u8 = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
            conf_heatmap = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
            conf_path = output_dir / f"{image_name}_confidence.png"
            cv2.imwrite(str(conf_path), conf_heatmap)

            # Severity map: Low (0.40-0.60), Medium (0.60-0.80), High (0.80+)
            severity_map = np.zeros((height, width), dtype=np.uint8)
            changed_region = change_mask > 0
            prob_changed = prob_map[changed_region]
            severity_map[changed_region] = 1  # Low
            if len(prob_changed) > 0:
                med_mask = changed_region & (prob_map >= 0.60)
                high_mask = changed_region & (prob_map >= 0.80)
                severity_map[med_mask] = 2  # Medium
                severity_map[high_mask] = 3  # High

            # Colorize severity: 0=black, 1=yellow, 2=orange, 3=red
            severity_color = np.zeros((height, width, 3), dtype=np.uint8)
            severity_color[severity_map == 1] = [0, 255, 255]   # Yellow (Low)
            severity_color[severity_map == 2] = [0, 165, 255]   # Orange (Medium)
            severity_color[severity_map == 3] = [0, 0, 255]     # Red (High)
            severity_path = output_dir / f"{image_name}_severity.png"
            cv2.imwrite(str(severity_path), severity_color)

            # Severity statistics
            low_px = int(np.sum(severity_map == 1))
            med_px = int(np.sum(severity_map == 2))
            high_px = int(np.sum(severity_map == 3))
            mean_conf = float(np.mean(prob_map[changed_region])) if changed_pixels > 0 else 0.0
            severity_stats = {
                "low_pixels": low_px,
                "low_percentage": round(low_px / max(total_pixels, 1) * 100, 2),
                "medium_pixels": med_px,
                "medium_percentage": round(med_px / max(total_pixels, 1) * 100, 2),
                "high_pixels": high_px,
                "high_percentage": round(high_px / max(total_pixels, 1) * 100, 2),
                "mean_confidence": round(mean_conf, 3),
                "max_confidence": round(float(prob_map.max()), 3),
            }
        else:
            conf_path = None
            severity_path = None
            severity_stats = {}

        # Object extraction uses a dilated version of the change mask so the
        # full extent of each changed object (its edges fall just outside the
        # raw mask) is covered. The raw mask stays authoritative for stats.
        obj_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        object_mask = cv2.dilate(change_mask, obj_kernel)

        if job_id:
            update_job(job_id, 50, "Running segmentation")
        self.logger.info("Running semantic segmentation")

        from src.loveda_segmenter import LoveDASegmenter
        segmenter = LoveDASegmenter()

        before_rgb, before_semantic, before_confidence = segmenter.predict(str(before_image_path))
        after_rgb, after_semantic, after_confidence = segmenter.predict(str(norm_after_path))

        before_semantic_path = output_dir / f"{image_name}_before_semantic.png"
        after_semantic_path = output_dir / f"{image_name}_after_semantic.png"
        colorize(before_semantic).save(str(before_semantic_path))
        colorize(after_semantic).save(str(after_semantic_path))

        # Land-cover transitions (forest -> barren, field -> building,
        # building -> background, ...) are real changes even when the
        # ChangeFormer mask misses them. Add spatially coherent transition
        # regions to the object mask so land-cover changes are classified
        # and reported. Small regions only qualify when they are a *clean*
        # switch (>=60% of the region flips to one class that was mostly
        # absent before) with real pixel evidence - this captures individual
        # new buildings while rejecting scattered texture noise on
        # AI-generated pairs.
        #
        # Evidence must come from the RAW pair.  The histogram-matched
        # after image is used by the segmenter, but its nonlinear L-channel
        # remap *manufactures* large pixel differences in regions whose
        # illumination differs between acquisitions (a region the change
        # detectors agree is unchanged can show matched-diff 113 vs the
        # global 55), so matched pixels are not admissible evidence.
        pixel_diff = np.abs(before_image.astype(np.float32) -
                            raw_after_image.astype(np.float32)).mean(axis=2)
        # Evidence is judged *relative* to the pair's global pixel
        # difference: on seasonally-different pairs the whole image has a
        # large diff (e.g. mean 60/255), so an absolute floor like 10 lets
        # every segmentation flip qualify.  A transition region counts as
        # real change only when its mean diff clearly exceeds the global
        # background level (>= 1.25x).
        global_diff = float(pixel_diff.mean())
        transition = ((before_semantic != after_semantic) &
                      ((before_semantic > 0) | (after_semantic > 0))).astype(np.uint8)
        transition = cv2.morphologyEx(
            transition, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        n_trans, labels_trans, stats_trans, _ = cv2.connectedComponentsWithStats(
            transition, connectivity=8
        )
        coherent_transition = np.zeros_like(transition)
        for i in range(1, n_trans):
            area = int(stats_trans[i, cv2.CC_STAT_AREA])
            if area >= 400:
                px = labels_trans == i
                after_vals = after_semantic[px]
                before_vals = before_semantic[px]
                hist_after = np.bincount(after_vals.ravel(), minlength=8)
                hist_before = np.bincount(before_vals.ravel(), minlength=8)
                dom_after = int(hist_after[1:].argmax()) + 1 if hist_after[1:].sum() > 0 else 0
                if dom_after == 0:
                    continue
                frac_after = hist_after[dom_after] / max(area, 1)
                frac_before = hist_before[dom_after] / max(area, 1)
                mean_diff = float(pixel_diff[px].mean())
                ratio = mean_diff / max(global_diff, 1.0)
                if area >= Config.TRANSITION_MIN_AREA:
                    # Large regions count as real changes when they flip
                    # coherently to one class with pixel evidence clearly
                    # above the global background difference.  Mixed
                    # regions with weak pixel difference (texture noise)
                    # are rejected.
                    if frac_after >= 0.50 and ratio >= 1.25:
                        coherent_transition[px] = 1
                else:
                    # Smaller regions must be a *clean* switch with real
                    # pixel evidence - this captures individual new
                    # buildings while rejecting scattered texture noise on
                    # AI-generated pairs.
                    if (frac_after >= 0.60 and frac_before <= 0.40 and
                            mean_diff >= Config.TRANSITION_MIN_DIFF and ratio >= 1.25):
                        coherent_transition[px] = 1
        object_mask = np.maximum(object_mask, coherent_transition).astype(np.uint8)

        if job_id:
            update_job(job_id, 70, "Analyzing changes")
        class_statistics = self.analyzer.analyze(before_semantic, after_semantic, change_mask)

        if job_id:
            update_job(job_id, 80, "Detecting objects")
        self.logger.info("Detecting objects")
        objects = self.object_detector.detect(
            after_semantic, after_confidence, change_mask=object_mask,
            before_semantic=before_semantic, class_names=self.class_names,
        )

        special_objects = self.special_detector.detect_all(after_semantic, after_image)

        for bridge in special_objects.get("bridges", []):
            bb = [bridge["start"][0], bridge["start"][1],
                  bridge["end"][0] - bridge["start"][0],
                  bridge["end"][1] - bridge["start"][1]]
            cov = _bbox_change_overlap(bb, change_mask)
            if Config.ONLY_CHANGED_OBJECTS and cov < 0.5 * Config.OBJECT_CHANGE_THRESHOLD:
                continue
            objects.append({
                "id": len(objects) + 1,
                "class_id": 8,
                "class_name": "bridge",
                "pixels": int(bridge.get("length", 0) * 3),
                "confidence": 0.50,
                "bbox": bb,
                "centroid": [(bridge["start"][0] + bridge["end"][0]) // 2,
                             (bridge["start"][1] + bridge["end"][1]) // 2],
                "polygon": None,
                "change_overlap": round(cov, 3),
                "status": "detected",
            })

        for border in special_objects.get("borders", []):
            bb = [border["start"][0], border["start"][1],
                  border["end"][0] - border["start"][0],
                  border["end"][1] - border["start"][1]]
            cov = _bbox_change_overlap(bb, change_mask)
            if Config.ONLY_CHANGED_OBJECTS and cov < 0.5 * Config.OBJECT_CHANGE_THRESHOLD:
                continue
            objects.append({
                "id": len(objects) + 1,
                "class_id": 9,
                "class_name": "border",
                "pixels": int(border.get("length", 0)),
                "confidence": 0.50,
                "bbox": bb,
                "centroid": [(border["start"][0] + border["end"][0]) // 2,
                             (border["start"][1] + border["end"][1]) // 2],
                "polygon": None,
                "change_overlap": round(cov, 3),
                "status": "detected",
            })

        for checkpost in special_objects.get("checkposts", []):
            cb = checkpost.get("bbox", [0, 0, 0, 0])
            cov = _bbox_change_overlap(cb, change_mask)
            if Config.ONLY_CHANGED_OBJECTS and cov < 0.5 * Config.OBJECT_CHANGE_THRESHOLD:
                continue
            objects.append({
                "id": len(objects) + 1,
                "class_id": 10,
                "class_name": "checkpost",
                "pixels": int(checkpost.get("area", 0)),
                "confidence": 0.50,
                "bbox": cb,
                "centroid": [cb[0] + cb[2] // 2, cb[1] + cb[3] // 2],
                "polygon": None,
                "change_overlap": round(cov, 3),
                "status": "detected",
            })

        for compound in special_objects.get("compounds", []):
            cb = compound.get("bbox", [0, 0, 0, 0])
            cov = _bbox_change_overlap(cb, change_mask)
            if Config.ONLY_CHANGED_OBJECTS and cov < 0.5 * Config.OBJECT_CHANGE_THRESHOLD:
                continue
            objects.append({
                "id": len(objects) + 1,
                "class_id": 16,
                "class_name": "compound",
                "pixels": int(compound.get("area", 0)),
                "confidence": 0.5 + 0.3 * compound.get("enclosure", 0.0),
                "bbox": cb,
                "centroid": compound.get("centroid", [cb[0] + cb[2] // 2, cb[1] + cb[3] // 2]),
                "polygon": None,
                "change_overlap": round(cov, 3),
                "status": "detected",
            })

        try:
            yolo_objects = self.model_manager.detect_objects(
                str(before_image_path), str(norm_after_path), change_mask
            )
            for o in yolo_objects:
                class_id = VEHICLE_IDS.get(o["class_name"], 20)
                bbox = o.get("bbox", [0, 0, 0, 0])
                objects.append({
                    "id": len(objects) + 1,
                    "class_id": class_id,
                    "class_name": o["class_name"],
                    "pixels": bbox[2] * bbox[3],
                    "confidence": o.get("confidence", 0.8),
                    "bbox": bbox,
                    "centroid": [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2],
                    "polygon": None,
                    "change_overlap": o.get("change_overlap", 0.0),
                    "status": o.get("status", "detected"),
                })
        except Exception as e:
            logger.warning(f"Object detection skipped: {e}")

        # Changed regions that no detected object covers still get a
        # generic "changed_area" marker: the semantic segmenter's domain
        # (LoveDA land cover) often has nothing to say about a region the
        # change detectors flag (e.g. a small structure appearing), and an
        # unmarked change region looks like a detection miss on the
        # overlay.  Cover each uncovered changed component so the overlay
        # always highlights where change happened.
        covered = np.zeros_like(object_mask)
        for obj in objects:
            try:
                x, y, w, h = [int(v) for v in obj["bbox"]]
            except (TypeError, ValueError):
                continue
            covered[max(y, 0):y + h, max(x, 0):x + w] = 1
        uncovered = ((object_mask > 0) & (covered == 0)).astype(np.uint8)
        n_un, labels_un, stats_un, _ = cv2.connectedComponentsWithStats(uncovered, connectivity=8)
        for i in range(1, n_un):
            area = int(stats_un[i, cv2.CC_STAT_AREA])
            if area < 200:
                continue
            x, y, w, h = (int(stats_un[i, cv2.CC_STAT_LEFT]),
                          int(stats_un[i, cv2.CC_STAT_TOP]),
                          int(stats_un[i, cv2.CC_STAT_WIDTH]),
                          int(stats_un[i, cv2.CC_STAT_HEIGHT]))
            comp = labels_un == i
            ys, xs = np.nonzero(comp)
            objects.append({
                "id": len(objects) + 1,
                "class_id": 21,
                "class_name": "changed_area",
                "pixels": area,
                "confidence": round(float(prob_map[comp].mean()), 3) if prob_map is not None else 0.5,
                "bbox": [x, y, w, h],
                "centroid": [int(xs.mean()), int(ys.mean())],
                "polygon": None,
                "change_overlap": 1.0,
                "status": "detected",
            })
        if n_un > 1:
            self.logger.info("Added %d changed_area marker(s)", n_un - 1)

        if Config.ONLY_CHANGED_OBJECTS:
            objects = [o for o in objects if o.get("status") != "unchanged"]

        for idx, obj in enumerate(objects, start=1):
            obj["id"] = idx

        appeared = sum(1 for o in objects if o.get("status") == "appeared")
        unchanged = sum(1 for o in objects if o.get("status") == "unchanged")
        removed = sum(1 for o in objects if o.get("status") == "removed")
        self.logger.info("Objects: %s appeared, %s unchanged, %s removed",
                         appeared, unchanged, removed)

        if bounds:
            try:
                from src.geo_export import write_geojson

                write_geojson(
                    output_dir / f"{image_name}_objects.geojson",
                    objects, bounds, {"width": width, "height": height},
                )
            except Exception as e:
                self.logger.warning("GeoJSON export skipped: %s", e)

        if job_id:
            update_job(job_id, 85, "Generating overlay")
        overlay_image = self.overlay.draw(after_image, objects)
        overlay_path = output_dir / f"{image_name}_overlay.png"
        self.overlay.save(overlay_image, str(overlay_path))

        if job_id:
            update_job(job_id, 90, "Generating chart")
        chart_path = output_dir / f"{image_name}_chart.png"
        create_class_chart(objects, str(chart_path))

        severity_chart_path = output_dir / f"{image_name}_severity_chart.png"
        if severity_stats:
            create_severity_chart(severity_stats, str(severity_chart_path))
        else:
            severity_chart_path = None

        processing_time = round(time.time() - start_time, 2)

        if job_id:
            update_job(job_id, 95, "Generating reports")
        report = self.report_generator.create(
            image_name=image_name,
            changed_pixels=changed_pixels,
            class_statistics=class_statistics,
            objects=objects,
            image_size={"width": width, "height": height},
            processing_time=processing_time,
        )
        report["change_detection"]["change_percentage"] = change_percentage

        json_path = output_dir / f"{image_name}.json"
        self.report_generator.save_json(report, str(json_path))

        pdf_path = output_dir / f"{image_name}.pdf"
        self.pdf_generator.create(
            output_path=str(pdf_path),
            before_image=str(before_image_path),
            after_image=str(after_image_path),
            overlay_image=str(overlay_path),
            change_percentage=change_percentage,
            statistics=class_statistics,
            objects=objects,
            before_semantic=str(before_semantic_path),
            after_semantic=str(after_semantic_path),
            change_mask=str(binary_path),
            chart_path=str(chart_path),
            processing_time=processing_time,
        )

        # Add severity and confidence to report
        report["change_detection"]["severity"] = severity_stats
        report["change_detection"]["confidence_map"] = (
            f"/static/outputs/{image_name}_confidence.png" if conf_path else None
        )
        report["change_detection"]["severity_map"] = (
            f"/static/outputs/{image_name}_severity.png" if severity_path else None
        )

        self.logger.info("Pipeline completed in %.2fs", processing_time)

        files = {
            "overlay": f"/static/outputs/{image_name}_overlay.png",
            "json": f"/static/outputs/{image_name}.json",
            "pdf": f"/static/outputs/{image_name}.pdf",
            "chart": f"/static/outputs/{image_name}_chart.png",
            "binary_mask": f"/static/outputs/{image_name}_binary_mask.png",
            "before_semantic": f"/static/outputs/{image_name}_before_semantic.png",
            "after_semantic": f"/static/outputs/{image_name}_after_semantic.png",
        }
        if conf_path:
            files["confidence_map"] = f"/static/outputs/{image_name}_confidence.png"
        if severity_path:
            files["severity_map"] = f"/static/outputs/{image_name}_severity.png"
        if severity_chart_path:
            files["severity_chart"] = f"/static/outputs/{image_name}_severity_chart.png"
        if bounds:
            files["geotiff_before"] = f"/static/outputs/{image_name}_before.tif"
            files["geotiff_after"] = f"/static/outputs/{image_name}_after.tif"
            files["geotiff_mask"] = f"/static/outputs/{image_name}_change_mask.tif"
            files["geojson"] = f"/static/outputs/{image_name}_objects.geojson"

        return {
            "success": True,
            "image_name": image_name,
            "image_size": {"width": width, "height": height},
            "processing_time": processing_time,
            "changed_pixels": changed_pixels,
            "change_percentage": change_percentage,
            "objects": objects,
            "statistics": class_statistics,
            "severity": severity_stats,
            "report": report,
            "files": files,
        }
