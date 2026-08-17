import logging

import cv2
import numpy as np

logger = logging.getLogger("satellite.overlay")


class OverlayGenerator:
    def __init__(self):
        self.shade_color = (255, 200, 0)
        self.border_color = (0, 255, 255)
        self.text_color = (255, 255, 255)
        self.status_shade = {
            "appeared": (0, 255, 0),
            "unchanged": (255, 200, 0),
            "removed": (0, 0, 255),
            "detected": (255, 0, 255),
        }
        self.status_border = {
            "appeared": (0, 255, 0),
            "unchanged": (0, 255, 255),
            "removed": (0, 0, 255),
            "detected": (255, 0, 255),
        }

    @staticmethod
    def _pts(obj, image_shape):
        polygon = obj.get("polygon")
        if polygon:
            return np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        x, y, w, h = obj.get("bbox", [0, 0, 0, 0])
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32).reshape(-1, 1, 2)

    def draw(self, image, objects):
        output = image.copy()
        height, width = image.shape[:2]
        shade_layer = np.zeros_like(image, dtype=np.uint8)
        mask_layer = np.zeros((height, width), dtype=np.uint8)

        for obj in objects:
            pts = self._pts(obj, image.shape)
            status = obj.get("status", "unchanged")
            color = self.status_shade.get(status, self.shade_color)
            cv2.fillPoly(mask_layer, [pts], 255)
            cv2.fillPoly(shade_layer, [pts], color)

        alpha = 0.25
        shaded = cv2.addWeighted(output, 1 - alpha, shade_layer, alpha, 0)
        mask_bool = mask_layer > 0
        output[mask_bool] = shaded[mask_bool]

        for obj in objects:
            pts = self._pts(obj, image.shape)
            status = obj.get("status", "unchanged")
            border_color = self.status_border.get(status, self.border_color)
            class_name = obj.get("class_name", "Unknown")
            confidence = obj.get("confidence", 0)
            object_id = obj.get("id", 0)

            cv2.polylines(output, [pts], True, border_color, 2)
            x, y, w, h = obj.get("bbox", [0, 0, 0, 0])
            label = f"#{object_id} {class_name} {confidence:.2f}"
            cv2.putText(output, label, (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.text_color, 2, cv2.LINE_AA)

        return output

    def save(self, image, path):
        success = cv2.imwrite(path, image)
        if not success:
            raise IOError(f"Failed saving overlay: {path}")
        logger.info("Overlay saved: %s", path)
