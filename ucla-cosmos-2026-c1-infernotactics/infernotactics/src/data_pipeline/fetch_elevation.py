"""
Pull USGS 3DEP elevation data for the Palisades Fire study area and save it
as a clipped GeoTIFF raster.

NOTE on implementation: the `py3dep` package (which wraps this same USGS 3DEP
WMS service) was tried first, but its async request path (pygeoogc ->
async_retriever) consistently returned corrupted/truncated TIFF bytes in
this environment (Python 3.14 + aiohttp), even after clearing its local
response cache. A plain synchronous WMS GetMap request via `owslib` against
the identical USGS endpoint returns a valid, correctly-sized GeoTIFF every
time, so we use that directly instead. This is the same real USGS 3DEP data
source (https://elevation.nationalmap.gov/.../3DEPElevation/ImageServer),
just fetched without the buggy async dependency chain.
"""

import os
import sys
import time

import numpy as np
import rasterio
from owslib.wms import WebMapService
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import (
    BBOX_LEFT_BOTTOM_RIGHT_TOP,
    DATA_DIR,
    ELEVATION_PATH,
    STUDY_AREA_NAME,
)

WMS_URL = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer"
DEM_LAYER = "3DEPElevation:None"  # the raw elevation ("DEM") layer
REQUEST_CRS = "EPSG:5070"  # USA Contiguous Albers Equal Area, meters
RESOLUTION_M = 10  # standard nationwide 3DEP DEM resolution
MAX_RETRIES = 3


def _fetch_dem_tiff_bytes(bbox_4326):
    west, south, east, north = bbox_4326
    transformer = Transformer.from_crs(4326, 5070, always_xy=True)
    xmin, ymin = transformer.transform(west, south)
    xmax, ymax = transformer.transform(east, north)

    width = int((xmax - xmin) / RESOLUTION_M)
    height = int((ymax - ymin) / RESOLUTION_M)
    print(f"Requesting {width}x{height} px DEM in {REQUEST_CRS} "
          f"({width * height:,} pixels)")

    wms = WebMapService(WMS_URL, version="1.3.0")

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = wms.getmap(
                layers=[DEM_LAYER],
                srs=REQUEST_CRS,
                bbox=(xmin, ymin, xmax, ymax),
                size=(width, height),
                format="image/tiff",
            )
            data = resp.read()
            # Validate it's actually a readable raster before trusting it.
            with rasterio.io.MemoryFile(data) as memfile:
                with memfile.open():
                    pass
            return data, (xmin, ymin, xmax, ymax), (width, height)
        except Exception as exc:  # noqa: BLE001 - want to retry+report any failure
            last_error = exc
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: {exc!r}")
            time.sleep(2 * attempt)

    raise RuntimeError(
        f"USGS 3DEP WMS request failed after {MAX_RETRIES} attempts"
    ) from last_error


def fetch_elevation():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Fetching USGS 3DEP elevation (DEM) for: {STUDY_AREA_NAME}")
    print(f"BBox (west, south, east, north): {BBOX_LEFT_BOTTOM_RIGHT_TOP}")
    print(f"Resolution: {RESOLUTION_M}m")

    data, (xmin, ymin, xmax, ymax), (width, height) = _fetch_dem_tiff_bytes(
        BBOX_LEFT_BOTTOM_RIGHT_TOP
    )
    print(f"Downloaded {len(data):,} bytes")

    with rasterio.io.MemoryFile(data) as memfile:
        with memfile.open() as src:
            arr = src.read(1)
            nodata = src.nodata
            profile = src.profile

    profile.update(driver="GTiff", compress="lzw")
    with rasterio.open(ELEVATION_PATH, "w", **profile) as dst:
        dst.write(arr, 1)

    print(f"Saved elevation raster to: {ELEVATION_PATH}")

    print("\n--- Sanity check stats ---")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"CRS: {profile['crs']}")
    print(f"Bounds (EPSG:5070): ({xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f})")
    valid = arr[arr != nodata] if nodata is not None else arr
    valid = valid[~np.isnan(valid)]
    print(f"Elevation (m): min={valid.min():.1f}, max={valid.max():.1f}, "
          f"mean={valid.mean():.1f}")
    print(f"Nodata/NaN pixel fraction: {1 - valid.size / arr.size:.4f}")

    return arr


if __name__ == "__main__":
    fetch_elevation()
