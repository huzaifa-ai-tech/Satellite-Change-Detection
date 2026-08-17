import json
import logging
from datetime import datetime

logger = logging.getLogger("satellite.reports")


class ReportGenerator:
    def __init__(self):
        self.version = "1.0"

    def create(self, image_name, changed_pixels, class_statistics, objects,
               image_size=None, processing_time=None, change_percentage=None,
               severity_stats=None):
        detected_classes = {}
        for obj in objects:
            name = obj.get("class_name", "Unknown")
            pixels = obj.get("pixels", 0)
            if name not in detected_classes:
                detected_classes[name] = {"objects": 0, "pixels": 0}
            detected_classes[name]["objects"] += 1
            detected_classes[name]["pixels"] += pixels

        class_distribution = []
        for name, data in sorted(detected_classes.items(), key=lambda x: x[1]["pixels"], reverse=True):
            class_distribution.append({"class": name, "object_count": data["objects"], "pixels": data["pixels"]})

        appeared = sum(1 for o in objects if o.get("status") == "appeared")
        unchanged = sum(1 for o in objects if o.get("status") == "unchanged")
        removed = sum(1 for o in objects if o.get("status") == "removed")
        detected = sum(1 for o in objects if o.get("status") == "detected")

        report = {
            "report_information": {
                "report_version": self.version,
                "image_name": image_name,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "processing": {
                "processing_time_seconds": processing_time,
                "image_size": image_size,
            },
            "change_detection": {
                "changed_pixels": int(changed_pixels),
                "change_percentage": change_percentage,
                "appeared_objects": appeared,
                "unchanged_objects": unchanged,
                "removed_objects": removed,
                "detected_objects": detected,
                "severity": severity_stats or {},
            },
            "semantic_changes": {
                "summary": class_statistics.get("summary", {}),
                "transitions": class_statistics.get("transitions", []),
            },
            "detected_classes": class_distribution,
            "object_summary": {
                "total_objects": len(objects),
                "classes_detected": len(class_distribution),
            },
            "objects": objects,
        }
        return report

    def save_json(self, report, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        logger.info("Saved JSON: %s", filename)
