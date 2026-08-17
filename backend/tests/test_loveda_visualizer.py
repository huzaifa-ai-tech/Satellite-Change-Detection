"""Tests for the LoveDA semantic mask colorizer."""

import numpy as np
import pytest

from src.loveda_visualizer import colorize
from src.loveda_colors import COLORS


def test_colorize_shape_and_dtype():
    mask = np.zeros((32, 32), dtype=np.uint8)
    image = colorize(mask)
    assert image.size == (32, 32)
    assert np.asarray(image).dtype == np.uint8


def test_colorize_known_classes_use_defined_colors():
    mask = np.zeros((8, 8), dtype=np.uint8)
    for cls, color in COLORS.items():
        mask[0, 0] = cls
        image = np.asarray(colorize(mask))
        assert tuple(image[0, 0]) == tuple(color), f"class {cls}"


def test_colorize_unknown_class_default_gray():
    mask = np.full((4, 4), 255, dtype=np.uint8)
    image = np.asarray(colorize(mask))
    assert tuple(image[0, 0]) == (128, 128, 128)


def test_colorize_none_raises():
    with pytest.raises(ValueError):
        colorize(None)


def test_colorize_1d_mask_raises():
    with pytest.raises(ValueError):
        colorize(np.zeros(16, dtype=np.uint8))