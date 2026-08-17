"""Tests for the pure helper functions in src.models (no model weights)."""

import numpy as np

from src.models import ModelManager, _bbox_change_overlap


def test_bbox_change_overlap_full():
    mask = np.ones((100, 100), dtype=np.uint8)
    assert _bbox_change_overlap([10, 10, 20, 20], mask) == 1.0


def test_bbox_change_overlap_none():
    mask = np.zeros((100, 100), dtype=np.uint8)
    assert _bbox_change_overlap([10, 10, 20, 20], mask) == 0.0


def test_bbox_change_overlap_partial():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # 100 of the 400 box pixels changed
    box = [10, 10, 20, 20]
    assert _bbox_change_overlap(box, mask) == 0.25


def test_bbox_change_overlap_none_mask():
    assert _bbox_change_overlap([10, 10, 20, 20], None) == 1.0


def test_bbox_change_overlap_out_of_bounds():
    mask = np.ones((10, 10), dtype=np.uint8)
    assert _bbox_change_overlap([100, 100, 20, 20], mask) == 0.0


def test_bbox_change_overlap_invalid_box():
    mask = np.ones((10, 10), dtype=np.uint8)
    assert _bbox_change_overlap([], mask) == 0.0


def test_same_vehicle():
    assert ModelManager._same_vehicle("car", "car")
    assert ModelManager._same_vehicle("car", "truck")
    assert ModelManager._same_vehicle("truck", "bus")
    assert not ModelManager._same_vehicle("car", "ship")
    assert not ModelManager._same_vehicle("ship", "aircraft")


def test_iou():
    a = [0, 0, 10, 10]
    b = [5, 5, 10, 10]
    iou = ModelManager._iou(a, b)
    assert 0.0 < iou < 1.0
    assert ModelManager._iou(a, a) == 1.0
    assert ModelManager._iou(a, [100, 100, 10, 10]) == 0.0


def test_nms_keeps_highest_confidence():
    objects = [
        {"class_name": "car", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        {"class_name": "truck", "confidence": 0.5, "bbox": [1, 1, 10, 10]},
        {"class_name": "ship", "confidence": 0.8, "bbox": [50, 50, 10, 10]},
    ]
    kept = ModelManager._nms(objects)
    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.9
    assert kept[1]["class_name"] == "ship"


def test_nms_empty():
    assert ModelManager._nms([]) == []