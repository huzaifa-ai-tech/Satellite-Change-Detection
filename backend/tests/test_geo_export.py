"""Tests for GeoTIFF / GeoJSON export (src.geo_export)."""

import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from src.geo_export import write_geojson, write_geotiff

BOUNDS = [-97.75, 30.25, -97.74, 30.26]  # west, south, east, north


def test_geotiff_roundtrip(tmp_path):
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 3:9] = 255

    path = tmp_path / "change_mask.tif"
    write_geotiff(path, mask, BOUNDS)

    with rasterio.open(path) as src:
        assert src.crs.to_string() == "EPSG:4326"
        assert src.width == 20 and src.height == 10
        assert src.count == 1
        data = src.read(1)
        assert data.shape == (10, 20)
        assert int(data.sum()) == 6 * 6 * 255
        # Bounds transform: west edge of the raster maps to BOUNDS[0].
        assert abs(src.transform.c - BOUNDS[0]) < 1e-9
        assert abs(src.transform.f - BOUNDS[3]) < 1e-9


def test_geotiff_rgb(tmp_path):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200

    path = tmp_path / "rgb.tif"
    write_geotiff(path, rgb, BOUNDS)

    with rasterio.open(path) as src:
        assert src.count == 3
        assert int(src.read(1).max()) == 200


def test_geojson_structure(tmp_path):
    objects = [
        {
            "id": 1,
            "class_name": "building",
            "confidence": 0.93,
            "bbox": [0, 0, 50, 40],  # top-left quadrant
            "centroid": [25, 20],
            "status": "appeared",
            "change_overlap": 1.0,
        },
        {
            "id": 2,
            "class_name": "car",
            "confidence": 0.5,
            "bbox": [50, 40, 50, 60],
            "centroid": [75, 70],
            "status": "unchanged",
            "change_overlap": 0.0,
        },
    ]
    path = tmp_path / "objects.geojson"
    write_geojson(path, objects, BOUNDS, {"width": 100, "height": 100})

    fc = json.loads(path.read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection"
    # 2 polygons + 2 centroids
    assert len(fc["features"]) == 4

    poly = fc["features"][0]
    assert poly["geometry"]["type"] == "Polygon"
    ring = poly["geometry"]["coordinates"][0]
    assert len(ring) == 5  # closed ring
    # Top-left corner (px 0,0) -> north-west corner of bounds.
    assert ring[0] == [BOUNDS[0], BOUNDS[3]]
    # Bottom-right corner (px 50,40) -> half-width, 40% height south.
    expected_lng = BOUNDS[0] + (50 / 100) * (BOUNDS[2] - BOUNDS[0])
    expected_lat = BOUNDS[3] - (40 / 100) * (BOUNDS[3] - BOUNDS[1])
    assert abs(ring[2][0] - expected_lng) < 1e-12
    assert abs(ring[2][1] - expected_lat) < 1e-12

    props = poly["properties"]
    assert props["class_name"] == "building"
    assert props["status"] == "appeared"
    assert props["confidence"] == 0.93

    centroid = fc["features"][1]
    assert centroid["geometry"]["type"] == "Point"
    assert centroid["properties"]["kind"] == "centroid"