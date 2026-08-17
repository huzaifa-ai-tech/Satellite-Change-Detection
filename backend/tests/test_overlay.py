"""Tests for the change overlay generator (src.overlay)."""

import numpy as np

from src.overlay import OverlayGenerator


def _image(height=64, width=64):
    return np.full((height, width, 3), 50, dtype=np.uint8)


def test_draw_without_objects_returns_copy():
    image = _image()
    overlay = OverlayGenerator().draw(image, [])
    assert overlay.shape == image.shape
    assert np.array_equal(overlay, image)


def test_draw_applied_object_changes_pixels():
    image = _image()
    obj = {
        "id": 1,
        "class_name": "building",
        "confidence": 0.9,
        "bbox": [10, 10, 20, 20],
        "status": "appeared",
    }
    overlay = OverlayGenerator().draw(image, [obj])
    assert overlay.shape == image.shape
    # Pixels inside the box must differ from the flat background.
    assert not np.array_equal(overlay[15, 15], image[15, 15])
    # Pixels far outside the box remain unchanged.
    assert np.array_equal(overlay[0, 0], image[0, 0])


def test_draw_polygon_object():
    image = _image()
    obj = {
        "id": 2,
        "class_name": "car",
        "confidence": 0.8,
        "polygon": [[10, 10], [30, 10], [30, 30], [10, 30]],
        "status": "detected",
    }
    overlay = OverlayGenerator().draw(image, [obj])
    assert overlay.shape == image.shape
    assert not np.array_equal(overlay[20, 20], image[20, 20])


def test_draw_bbox_and_polygon_prefers_polygon():
    image = _image()
    obj = {
        "id": 3,
        "class_name": "truck",
        "confidence": 0.7,
        "bbox": [0, 0, 64, 64],
        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "status": "removed",
    }
    overlay = OverlayGenerator().draw(image, [obj])
    assert overlay.shape == image.shape
    # Inside the polygon -> shaded; outside it (but inside the bbox) -> not.
    assert not np.array_equal(overlay[5, 5], image[5, 5])
    assert np.array_equal(overlay[50, 50], image[50, 50])