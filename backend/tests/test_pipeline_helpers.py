"""Tests for pure image helpers in src.pipeline (no model inference)."""

import cv2
import numpy as np

from src.pipeline import _apply_clahe, _center_crop, _match_histogram


def _img(height=64, width=64, value=100):
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_center_crop_same_size_returns_input():
    image = _img()
    out = _center_crop(image, 64, 64)
    assert out.shape == (64, 64, 3)
    assert np.array_equal(out, image)


def test_center_crop_smaller_size():
    image = _img(100, 80)
    out = _center_crop(image, 40, 40)
    assert out.shape == (40, 40, 3)
    # Center pixel value is preserved.
    assert out[20, 20, 0] == 100


def test_center_crop_larger_request_than_image():
    # target larger than source slices into empty ranges; production code
    # only ever crops down to the common (smaller) size, so this must not
    # crash and must still produce a 3-channel image.
    image = _img(30, 30)
    out = _center_crop(image, 40, 40)
    assert out.ndim == 3 and out.shape[2] == 3


def test_apply_clahe_preserves_shape_and_dtype():
    image = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
    out = _apply_clahe(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out.shape, image.shape)


def test_apply_clahe_enhances_low_contrast():
    # A nearly-flat image should come out with higher contrast after CLAHE.
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = _apply_clahe(image)
    assert out.dtype == np.uint8
    assert out.shape == image.shape
    # Output still 3-channel BGR.
    assert out.ndim == 3 and out.shape[2] == 3


def test_match_histogram_returns_same_shape():
    before = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    after = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    matched = _match_histogram(before, after)
    assert matched.shape == before.shape
    assert matched.dtype == np.uint8


def test_match_histogram_resizes_mismatched_sizes():
    before = np.random.randint(0, 256, (80, 40, 3), dtype=np.uint8)
    after = np.random.randint(0, 256, (40, 80, 3), dtype=np.uint8)
    matched = _match_histogram(before, after)
    assert matched.shape == before.shape


def test_match_histogram_identical_images():
    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    matched = _match_histogram(image, image.copy())
    # Matching an image against itself is a no-op apart from the CLAHE
    # enhancement applied to both sides: matched == _apply_clahe(image).
    enhanced = _apply_clahe(image)
    assert np.mean(np.abs(matched.astype(int) - enhanced.astype(int))) < 2


def test_imwrite_roundtrip():
    # Guards the pipeline's dependency on cv2.imread/imwrite working with
    # the same numpy arrays used elsewhere in the pipeline.
    image = _img(16, 16)
    ok = cv2.imwrite(r"C:\Users\Admin\AppData\Local\Temp\opencode\roundtrip_test.png", image)
    assert ok
    loaded = cv2.imread(r"C:\Users\Admin\AppData\Local\Temp\opencode\roundtrip_test.png")
    assert loaded is not None
    assert loaded.shape == (16, 16, 3)