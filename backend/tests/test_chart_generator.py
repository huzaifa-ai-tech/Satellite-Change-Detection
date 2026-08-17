"""Tests for chart generation (palette handling, edge cases)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chart_generator import create_class_chart


def test_class_chart_more_classes_than_palette(tmp_path):
    # Regression: colors[:len(labels)] crashed with a matplotlib ValueError
    # as soon as an analysis produced more than 7 distinct object classes,
    # killing the whole job at the chart step.
    objects = [
        {"class_name": f"class_{i}", "confidence": 0.5}
        for i in range(12)
    ]
    out = tmp_path / "chart.png"
    create_class_chart(objects, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_class_chart_empty_objects(tmp_path):
    out = tmp_path / "chart_empty.png"
    create_class_chart([], str(out))
    assert out.exists()
    assert out.stat().st_size > 0