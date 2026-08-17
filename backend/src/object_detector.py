import logging

import cv2
import numpy as np

logger = logging.getLogger("satellite.objects")


class ObjectDetector:
    """Detects objects from semantic segmentation masks."""

    def __init__(self, min_pixels=200, min_confidence=0.15, max_fraction=0.40):
        self.min_pixels = min_pixels
        self.min_confidence = min_confidence
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.max_fraction = max_fraction
        self.priority = {
            "building": 1, "road": 2, "water": 3, "forest": 4,
            "agriculture": 5, "barren_land": 6, "other_land": 7,
            "bridge": 8, "border": 9, "checkpost": 10,
        }
        # Land-cover background classes are only reported when they form a
        # substantial region; tiny specks of water/road/forest are noise.
        self.background_min_pixels = 200

    def detect(self, semantic_mask, confidence_mask, change_mask=None, class_names=None,
               before_semantic=None):
        """Change-first object detection.

        The change mask is the ONLY source of truth: pixels outside it are
        never classified or reported. Inside changed regions, objects are
        the connected components of each land-cover class in the AFTER
        segmentation, so every individual building/field/etc. inside a
        changed area is reported separately (a big changed region holding
        several buildings yields several objects). Regions where the class
        disappeared (after = background) are reported as removed using the
        BEFORE class.
        """
        if change_mask is None:
            return []

        mask = (change_mask > 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        objects = []
        object_id = 1

        def _add_objects(class_id, px, status):
            nonlocal object_id
            class_name = class_names.get(class_id, "Unknown")
            count, labels, stats, _ = cv2.connectedComponentsWithStats(px.astype(np.uint8), connectivity=8)
            for i in range(1, count):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < self.min_pixels:
                    continue
                if class_name.lower() in self.priority and class_name.lower() != "building":
                    if area < self.background_min_pixels:
                        continue

                comp = labels == i
                confidence = float(np.mean(confidence_mask[comp]))
                if confidence < self.min_confidence:
                    continue

                semantic_change = 0.0
                if before_semantic is not None:
                    semantic_change = float(np.mean(before_semantic[comp] != class_id))

                ys, xs = np.nonzero(comp)
                x, y = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                w, h = x2 - x + 1, y2 - y + 1
                if w < 3 or h < 3:
                    continue
                img_h, img_w = semantic_mask.shape[:2]
                if (w * h) > self.max_fraction * img_h * img_w:
                    continue

                cx, cy = float(np.mean(xs)), float(np.mean(ys))
                component = comp.astype(np.uint8)
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygon = None
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.002 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    polygon = approx.reshape(-1, 2).tolist()

                objects.append({
                    "id": object_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "pixels": area,
                    "confidence": round(confidence, 3),
                    "bbox": [x, y, w, h],
                    "centroid": [int(cx), int(cy)],
                    "polygon": polygon,
                    "change_overlap": 1.0,
                    "semantic_change": round(semantic_change, 3),
                    "status": status,
                })
                object_id += 1

        # Appeared / changed: connected components of each AFTER class
        # inside the changed regions.
        for class_id in np.unique(semantic_mask):
            class_id = int(class_id)
            if class_id == 0:
                continue
            inside = (semantic_mask == class_id) & (mask > 0)
            if inside.sum() < self.min_pixels:
                continue
            _add_objects(class_id, inside, "appeared")

        # Removed: regions where the AFTER image became background but the
        # BEFORE image had a class there.
        if before_semantic is not None:
            bg_after = (semantic_mask == 0) & (mask > 0)
            for class_id in np.unique(before_semantic):
                class_id = int(class_id)
                if class_id == 0:
                    continue
                gone = (before_semantic == class_id) & bg_after
                if gone.sum() < self.min_pixels:
                    continue
                _add_objects(class_id, gone, "removed")

        objects.sort(key=lambda x: (self.priority.get(x["class_name"].lower(), 99), -x["pixels"]))
        for idx, obj in enumerate(objects, start=1):
            obj["id"] = idx

        logger.info("Objects detected: %s", len(objects))
        return objects


class SpecialObjectDetector:
    """Detects bridges, borders, and checkposts using structural analysis."""

    def __init__(self):
        self.min_bridge_length = 15
        self.min_bridge_width = 2
        self.max_bridge_width = 20
        self.min_border_length = 20
        self.checkpost_min_area = 15
        self.checkpost_max_area = 300

    def detect_bridges(self, semantic_mask, rgb_image=None):
        """Detect bridges as linear structures over water or connecting land."""
        bridges = []
        water_mask = np.isin(semantic_mask, [3]).astype(np.uint8) * 255
        land_mask = np.isin(semantic_mask, [1, 2, 4, 5, 6]).astype(np.uint8) * 255

        edges = cv2.Canny(land_mask, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                                 minLineLength=self.min_bridge_length,
                                 maxLineGap=10)

        if lines is not None:
            for line in lines:
                if isinstance(line, (list, tuple)) and len(line) > 0:
                    coords = line[0] if isinstance(line[0], (list, tuple, np.ndarray)) else line
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length >= self.min_bridge_length:
                            bridges.append({
                                "type": "bridge",
                                "start": [x1, y1],
                                "end": [x2, y2],
                                "length": float(length),
                            })

        if rgb_image is not None and len(bridges) == 0:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                     minLineLength=self.min_bridge_length,
                                     maxLineGap=5)
            if lines is not None:
                for line in lines:
                    if isinstance(line, (list, tuple)) and len(line) > 0:
                        coords = line[0] if isinstance(line[0], (list, tuple, np.ndarray)) else line
                        if len(coords) >= 4:
                            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                            if length >= self.min_bridge_length:
                                bridges.append({
                                    "type": "bridge",
                                    "start": [x1, y1],
                                    "end": [x2, y2],
                                    "length": float(length),
                                })

        return bridges

    def detect_borders(self, semantic_mask, rgb_image=None):
        """Detect borders as boundaries between different land cover types."""
        borders = []

        edges = cv2.Canny(semantic_mask.astype(np.uint8) * 30, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                 minLineLength=self.min_border_length,
                                 maxLineGap=5)

        if lines is not None:
            for line in lines:
                if isinstance(line, (list, tuple)) and len(line) > 0:
                    coords = line[0] if isinstance(line[0], (list, tuple, np.ndarray)) else line
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length >= self.min_border_length:
                            borders.append({
                                "type": "border",
                                "start": [x1, y1],
                                "end": [x2, y2],
                                "length": float(length),
                            })

        return borders

    def detect_checkposts(self, semantic_mask, rgb_image=None):
        """Detect checkposts as very small rectangular structures near roads.
        Only detects if confidence is high enough to avoid false positives."""
        checkposts = []

        building_mask = np.isin(semantic_mask, [1]).astype(np.uint8) * 255
        road_mask = np.isin(semantic_mask, [2]).astype(np.uint8) * 255

        road_dilated = cv2.dilate(road_mask, np.ones((20, 20), np.uint8), iterations=1)

        contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 8 <= area <= 80:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                solidity = area / (w * h) if w * h > 0 else 0

                if 0.7 <= aspect_ratio <= 1.5 and solidity > 0.7:
                    cx, cy = x + w // 2, y + h // 2
                    h_img, w_img = road_dilated.shape[:2]
                    if 0 <= cy < h_img and 0 <= cx < w_img:
                        if road_dilated[cy, cx] > 0:
                            checkposts.append({
                                "type": "checkpost",
                                "bbox": [int(x), int(y), int(w), int(h)],
                                "area": int(area),
                            })

        return checkposts

    def detect_compounds(self, semantic_mask, rgb_image=None):
        """Detect compounds: walled/enclosed clusters of buildings
        (schools, factories, housing colonies, barracks, etc.)."""
        compounds = []
        building_mask = np.isin(semantic_mask, [1]).astype(np.uint8)
        road_mask = np.isin(semantic_mask, [2]).astype(np.uint8)

        # Merge nearby buildings into compound candidates
        merge_kernel = np.ones((9, 9), np.uint8)
        merged = cv2.dilate(building_mask, merge_kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

        h_img, w_img = semantic_mask.shape

        for i in range(1, num_labels):
            x, y, w, h = (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            )
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 400 or area > 0.5 * (h_img * w_img):
                continue
            if w < 15 or h < 15:
                continue

            # Require at least 2 distinct buildings inside the candidate
            component = (labels == i).astype(np.uint8)
            buildings_inside = building_mask[component > 0]
            num_building = int(np.count_nonzero(buildings_inside))
            if num_building < 2:
                continue

            # Count separate building blobs for strength scoring
            bg_build = building_mask * component
            inside_count, inside_labels, _, _ = cv2.connectedComponentsWithStats(
                (bg_build > 0).astype(np.uint8), connectivity=8
            )
            blob_count = inside_count - 1
            if blob_count < 2:
                continue

            # Enclosure score: fraction of the border band (straddling the
            # candidate edge) covered by road — compounds are typically
            # ringed by streets or walls
            band = 3
            y0, y1 = max(y - band, 0), min(y + h + band, h_img)
            x0, x1 = max(x - band, 0), min(x + w + band, w_img)
            region_road = road_mask[y0:y1, x0:x1]
            inner_h, inner_w = h, w
            ring = np.zeros_like(region_road, dtype=bool)
            ring[:band, :] = True
            ring[-band:, :] = True
            ring[:, :band] = True
            ring[:, -band:] = True
            ring[band:inner_h + band, band:inner_w + band] = False
            perimeter_px = max(2 * (w + h) + 4, 1)
            road_on_ring = int(np.count_nonzero(region_road[ring]))
            enclosure = road_on_ring / perimeter_px

            if enclosure < 0.05 and blob_count < 3:
                continue

            cx, cy = x + w // 2, y + h // 2
            compounds.append({
                "type": "compound",
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(area),
                "centroid": [int(cx), int(cy)],
                "building_count": int(blob_count),
                "enclosure": round(float(enclosure), 3),
            })

        return compounds

    def detect_all(self, semantic_mask, rgb_image=None):
        """Run all special object detectors."""
        results = {"bridges": [], "borders": [], "checkposts": [], "compounds": []}
        try:
            results["bridges"] = self.detect_bridges(semantic_mask, rgb_image)
        except Exception as e:
            logger.debug("Bridge detection: %s", e)
        try:
            results["borders"] = self.detect_borders(semantic_mask, rgb_image)
        except Exception as e:
            logger.debug("Border detection: %s", e)
        try:
            results["checkposts"] = self.detect_checkposts(semantic_mask, rgb_image)
        except Exception as e:
            logger.debug("Checkpost detection: %s", e)
        try:
            results["compounds"] = self.detect_compounds(semantic_mask, rgb_image)
        except Exception as e:
            logger.debug("Compound detection: %s", e)
        return results
