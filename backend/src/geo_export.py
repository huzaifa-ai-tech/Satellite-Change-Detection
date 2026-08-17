"""GeoTIFF / GeoJSON export of analysis results.

The pipeline optionally receives geographic ``bounds`` (``[west, south,
east, north]`` in EPSG:4326).  When present it writes:

* ``{name}_before.tif`` / ``{name}_after.tif`` — RGB composites
* ``{name}_change_mask.tif`` — binary change mask
* ``{name}_objects.geojson`` — detected objects as geo-referenced
  polygons (plus point centroids), so results can be opened in QGIS,
  ArcGIS, Google Earth, etc.
"""

import json
import logging

import numpy as np

logger = logging.getLogger("satellite")

DEFAULT_CRS = "EPSG:4326"


def write_geotiff(path, array, bounds, crs=DEFAULT_CRS):
    """Write a 2D (single band) or 3D (H, W, bands) array as a GeoTIFF.

    ``bounds`` is ``[west, south, east, north]`` in degrees.
    """
    import rasterio
    from rasterio.transform import from_bounds

    array = np.asarray(array)
    if array.ndim == 2:
        height, width = array.shape
        count = 1
        data = array.astype("uint8")
    elif array.ndim == 3:
        height, width, bands = array.shape
        count = bands
        data = array.astype("uint8")
    else:
        raise ValueError(f"Unsupported array shape for GeoTIFF: {array.shape}")

    west, south, east, north = bounds
    transform = from_bounds(west, south, east, north, width, height)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width,
        count=count, dtype="uint8", crs=crs, transform=transform,
    ) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            for band in range(count):
                dst.write(data[:, :, band], band + 1)
    logger.info("Wrote GeoTIFF %s (%dx%d, %d band(s))", path, width, height, count)
    return str(path)


def write_geojson(path, objects, bounds, image_size):
    """Write detected objects as a geo-referenced GeoJSON FeatureCollection.

    Objects are given in pixel coordinates; they are projected into the
    ``bounds`` rectangle (top-left -> north-west corner).
    """
    width = int(image_size.get("width", 0))
    height = int(image_size.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image_size for GeoJSON: {image_size}")

    west, south, east, north = bounds

    def to_lng(px):
        return west + (px / width) * (east - west)

    def to_lat(py):
        return north - (py / height) * (north - south)

    features = []
    for obj in objects:
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        x, y, w, h = bbox
        polygon = [
            [to_lng(x), to_lat(y)],
            [to_lng(x + w), to_lat(y)],
            [to_lng(x + w), to_lat(y + h)],
            [to_lng(x), to_lat(y + h)],
            [to_lng(x), to_lat(y)],
        ]
        properties = {
            "id": obj.get("id"),
            "class_name": obj.get("class_name"),
            "confidence": round(float(obj.get("confidence", 0)), 4),
            "status": obj.get("status"),
            "change_overlap": round(float(obj.get("change_overlap", 0)), 4),
        }
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [polygon]},
            "properties": properties,
        })

        centroid = obj.get("centroid")
        if centroid and len(centroid) == 2:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [to_lng(centroid[0]), to_lat(centroid[1])],
                },
                "properties": {**properties, "kind": "centroid"},
            })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"type": "FeatureCollection", "features": features},
            f, indent=2,
        )
    logger.info("Wrote GeoJSON %s (%d features)", path, len(features))
    return str(path)