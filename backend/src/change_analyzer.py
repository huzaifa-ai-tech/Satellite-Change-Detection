import logging

import numpy as np

logger = logging.getLogger("satellite.analyzer")


class ChangeAnalyzer:
    IGNORE_CLASSES = {0: "Ignore"}

    def __init__(self, class_names):
        self.class_names = class_names or {}

    def analyze(self, before_mask, after_mask, change_mask):
        changed_pixels_mask = change_mask > 0
        before = before_mask[changed_pixels_mask]
        after = after_mask[changed_pixels_mask]
        total_changed = len(before)

        if total_changed == 0:
            return {"summary": {"total_changed_pixels": 0, "total_transitions": 0, "major_change": "None"}, "transitions": []}

        transitions = {}
        for b, a in zip(before, after):
            b, a = int(b), int(a)
            if b == a or b in self.IGNORE_CLASSES or a in self.IGNORE_CLASSES:
                continue
            before_name = self.class_names.get(b, "Unknown")
            after_name = self.class_names.get(a, "Unknown")
            key = f"{before_name} -> {after_name}"
            if key not in transitions:
                transitions[key] = {"from": before_name, "to": after_name, "pixels": 0}
            transitions[key]["pixels"] += 1

        ranked = []
        for key, value in transitions.items():
            percentage = (value["pixels"] / total_changed) * 100
            if percentage < 0.05:
                continue
            value["percentage"] = round(percentage, 2)
            value["severity"] = "High" if percentage >= 25 else "Medium" if percentage >= 10 else "Low"
            ranked.append(value)

        ranked.sort(key=lambda x: x["pixels"], reverse=True)
        logger.info("Semantic transitions: %s", len(ranked))

        return {
            "summary": {
                "total_changed_pixels": int(total_changed),
                "total_transitions": len(ranked),
                "major_change": f"{ranked[0]['from']} -> {ranked[0]['to']}" if ranked else "None",
            },
            "transitions": ranked,
        }
