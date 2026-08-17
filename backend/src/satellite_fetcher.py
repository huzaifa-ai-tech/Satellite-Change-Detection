import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

logger = logging.getLogger("satellite.fetcher")

STAC_API = "https://earth-search.aws.element84.com/v1"
SENTINEL_COLLECTION = "sentinel-2-l2a"
MAX_CLOUD = 20
TARGET_SIZE = (512, 512)
MIN_TARGET_SIZE = 256
MAX_TARGET_SIZE = 1536
SENTINEL_GROUND_RES_M = 10.0
SPECTRAL_CHANGE_THRESHOLD = float(os.getenv("SPECTRAL_CHANGE_THRESHOLD", "0.15"))


def _target_size_for_bbox(bbox):
    """Pick output pixel size matching Sentinel-2's native 10m ground resolution.

    A region smaller than ~2.5 km should NOT be upscaled to 512x512 (that is what
    made the imagery look blurry). Compute pixels so that 1 px ~= 10 m, clamped to
    a sensible range for the CPU pipeline.
    """
    west, south, east, north = bbox
    lat = (south + north) / 2
    meters_per_deg_lat = 111320.0
    meters_per_deg_lng = 111320.0 * math.cos(math.radians(lat))

    width = max((east - west) * meters_per_deg_lng / SENTINEL_GROUND_RES_M, MIN_TARGET_SIZE)
    height = max((north - south) * meters_per_deg_lat / SENTINEL_GROUND_RES_M, MIN_TARGET_SIZE)

    scale = min(1.0, MAX_TARGET_SIZE / max(width, height))
    return (int(round(width * scale)), int(round(height * scale)))


def search_sentinel_imagery(bbox, date_start, date_end, max_cloud=MAX_CLOUD, limit=5):
    params = {
        "collections": json.dumps([SENTINEL_COLLECTION]),
        "bbox": ",".join(str(x) for x in bbox),
        "datetime": f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        "limit": limit,
    }
    if max_cloud < 100:
        params["query"] = json.dumps({"eo:cloud_cover": {"lte": max_cloud}})
    resp = requests.get(f"{STAC_API}/search", params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])

    results = []
    for feat in features:
        props = feat["properties"]
        results.append({
            "id": feat["id"],
            "datetime": props.get("datetime"),
            "cloud_cover": props.get("eo:cloud_cover", 100),
            "assets": feat["assets"],
            "bbox": feat.get("bbox"),
        })
    results.sort(key=lambda x: x.get("cloud_cover", 100))
    return results


def download_visual_image(assets, bbox, target_size=TARGET_SIZE):
    """Download the visual (RGB) preview from Sentinel-2 assets."""
    visual_key = None
    for key in ["visual", "rendered_preview"]:
        if key in assets:
            visual_key = key
            break
    if visual_key is None:
        for key in ["true_color", "true-color"]:
            if key in assets:
                visual_key = key
                break

    # If no visual, combine RGB bands
    if visual_key is None and all(b in assets for b in ["red", "green", "blue"]):
        return _download_rgb_composite(assets, bbox, target_size)

    if visual_key is None:
        raise ValueError("No visual or RGB assets available")

    url = assets[visual_key]["href"]
    return _download_cog_region(url, bbox, target_size)


def _download_cog_region(url, bbox, target_size):
    """Download a region of interest from a COG."""
    try:
        import rasterio
        from rasterio.windows import Window
        from shapely.geometry import box
    except ImportError:
        return _download_full_image(url, target_size)

    try:
        with rasterio.open(url) as src:
            west, south, east, north = bbox
            src_crs = src.crs

            if src_crs and src_crs.to_string() != "EPSG:4326":
                from rasterio.warp import transform_bounds
                try:
                    west, south, east, north = transform_bounds(
                        "EPSG:4326", src_crs, west, south, east, north
                    )
                except Exception:
                    pass

            window = src.window(west, south, east, north)
            window = window.round_lengths().round_offsets()

            out_shape = (target_size[1], target_size[0])
            data = src.read(window=window, out_shape=out_shape)
            data = data[:3] if data.shape[0] >= 3 else data

            data = np.clip(data, 0, None)
            if data.max() > 0:
                data = (data / data.max() * 255).astype(np.uint8)
            else:
                data = data.astype(np.uint8)

            img = np.transpose(data, (1, 2, 0))
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning("COG download failed: %s", e)
        return _download_full_image(url, target_size)


def _download_full_image(url, target_size=TARGET_SIZE):
    """Fallback: download full image and resize."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _download_rgb_composite(assets, bbox, target_size):
    """Combine red, green, blue bands into a color image."""
    bands = {}
    for color, band_key in [("red", "red"), ("green", "green"), ("blue", "blue")]:
        url = assets[band_key]["href"]
        arr = _download_cog_region_full(url, bbox, target_size)
        if arr is None:
            return None
        bands[color] = arr

    rgb = np.stack([bands["red"], bands["green"], bands["blue"]], axis=-1)
    rgb = np.clip(rgb, 0, None)
    if rgb.max() > 0:
        rgb = (rgb / rgb.max() * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _shift_date(date_str, days):
    """Return date shifted by a number of days (e.g. '2024-01-01', -90)."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (d + timedelta(days=days)).strftime("%Y-%m-%d")


def _nearest_scene(items, target_date):
    """Pick the scene whose acquisition date is closest to the target date."""
    target = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def distance(item):
        dt = datetime.fromisoformat(item["datetime"].replace("Z", "+00:00"))
        return abs((dt - target).total_seconds())

    return min(items, key=distance)


def _download_cog_region_full(url, bbox, target_size):
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(url) as src:
            west, south, east, north = bbox
            if src.crs and src.crs.to_string() != "EPSG:4326":
                west, south, east, north = transform_bounds(
                    "EPSG:4326", src.crs, west, south, east, north
                )
            window = src.window(west, south, east, north)
            window = window.round_lengths().round_offsets()
            out_shape = (target_size[1], target_size[0])
            return src.read(1, window=window, out_shape=out_shape)
    except Exception:
        return None


def _band_float(assets, key, bbox, target_size):
    if key not in assets:
        return None
    arr = _download_cog_region_full(assets[key]["href"], bbox, target_size)
    if arr is None:
        return None
    return np.clip(np.asarray(arr, dtype=np.float32), 0, None)


def _ndvi(assets, bbox, target_size):
    red = _band_float(assets, "red", bbox, target_size)
    nir = _band_float(assets, "nir", bbox, target_size)
    if red is None or nir is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nir - red) / (nir + red + 1e-6)


def compute_spectral_change_mask(before_assets, after_assets, bbox, target_size, threshold=None):
    """NDVI-difference change mask for Sentinel-2 (works at 10m ground resolution).

    The ChangeFormer model is trained on high-res aerial imagery (LEVIR) and does
    not respond to 10m Sentinel-2 data, so we additionally flag real spectral
    changes (vegetation loss/gain, land clearing, water) via NDVI differencing.
    """
    if threshold is None:
        threshold = SPECTRAL_CHANGE_THRESHOLD
    ndvi_before = _ndvi(before_assets, bbox, target_size)
    ndvi_after = _ndvi(after_assets, bbox, target_size)
    if ndvi_before is None or ndvi_after is None:
        logger.warning("Spectral change detection unavailable (missing red/nir bands)")
        return None

    diff = np.abs(ndvi_after - ndvi_before)
    mask = (diff > threshold).astype(np.uint8)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    logger.info("Spectral change pixels: %s (%.2f%%)",
                int(np.sum(mask)), round(100 * np.sum(mask) / mask.size, 2))
    return mask


def _find_scene(bbox, start, end, target_date, label):
    """Find the scene closest to target_date, widening window and relaxing clouds as needed."""
    for extra in (0, 90):
        for cloud in (MAX_CLOUD, 100):
            items = search_sentinel_imagery(
                bbox, _shift_date(start, -extra), _shift_date(end, extra),
                max_cloud=cloud, limit=20,
            )
            if items:
                return _nearest_scene(items, target_date)
    raise ValueError(f"No suitable imagery found for {label}")


def fetch_satellite_pair(lat, lng, buffer_deg, date1, date2, target_size=TARGET_SIZE):
    """Fetch a pair of satellite images for change detection.

    The "before" image is taken near date1 (strictly before date2) and the
    "after" image is taken near date2 (on or after date2), so the two scenes
    always come from different dates.

    Args:
        lat, lng: Center coordinates in degrees
        buffer_deg: Buffer around center in degrees
        date1: Date for the "before" image (e.g. '2024-01-01')
        date2: Date for the "after" image (e.g. '2025-01-01')

    Returns:
        (before_img_bgr, after_img_bgr, metadata, spectral_change_mask)
    """
    bbox = [lng - buffer_deg, lat - buffer_deg, lng + buffer_deg, lat + buffer_deg]

    if date1 > date2:
        date1, date2 = date2, date1

    target_size = _target_size_for_bbox(bbox)
    logger.info("Fetching imagery at native resolution, target size: %s", target_size)

    before = _find_scene(bbox, _shift_date(date1, -90), _shift_date(date2, -1), date1, f"period {date1} (before)")
    after = _find_scene(bbox, date2, _shift_date(date2, 90), date2, f"period {date2} (after)")

    logger.info("Before image: %s (cloud: %s%%)", before["id"], before.get("cloud_cover"))
    logger.info("After image: %s (cloud: %s%%)", after["id"], after.get("cloud_cover"))

    before_img = download_visual_image(before["assets"], bbox, target_size)
    after_img = download_visual_image(after["assets"], bbox, target_size)

    spectral_change_mask = compute_spectral_change_mask(
        before["assets"], after["assets"], bbox, target_size
    )

    metadata = {
        "bounds": bbox,
        "crs": "EPSG:4326",
        "before": {
            "id": before["id"],
            "datetime": before["datetime"],
            "cloud_cover": before["cloud_cover"],
        },
        "after": {
            "id": after["id"],
            "datetime": after["datetime"],
            "cloud_cover": after["cloud_cover"],
        },
    }

    return before_img, after_img, metadata, spectral_change_mask
