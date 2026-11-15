"""
Pull building footprint + height data for the Palisades Fire study area from
the LA GeoHub "Building Footprints" dataset (LARIAC4), clip to the bounding
box, and save locally.

Source dataset page:
  https://geohub.lacity.org/datasets/813fcefde1f64b209103107b26a8909f_0
Resolved via the ArcGIS item API to a public, unauthenticated FeatureServer:
  https://services5.arcgis.com/7nsPwEMP38bSkCjy/arcgis/rest/services/Building_Footprints/FeatureServer/0

The service caps each query at maxRecordCount=1000 records, so we page
through results with resultOffset until exhausted.
"""

import os
import sys

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import (
    BBOX_LEFT_BOTTOM_RIGHT_TOP,
    BUILDINGS_PATH,
    DATA_DIR,
    EAST,
    NORTH,
    SOUTH,
    STUDY_AREA_NAME,
    WEST,
)

FEATURE_SERVER_QUERY_URL = (
    "https://services5.arcgis.com/7nsPwEMP38bSkCjy/arcgis/rest/services/"
    "Building_Footprints/FeatureServer/0/query"
)
PAGE_SIZE = 1000


def _bbox_envelope_json():
    return (
        f'{{"xmin":{WEST},"ymin":{SOUTH},"xmax":{EAST},"ymax":{NORTH}}}'
    )


def _get_count():
    params = {
        "where": "1=1",
        "geometry": _bbox_envelope_json(),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "returnCountOnly": "true",
        "f": "json",
    }
    resp = requests.get(FEATURE_SERVER_QUERY_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"ArcGIS API error: {data['error']}")
    return data["count"]


def _fetch_page(offset):
    params = {
        "where": "1=1",
        "geometry": _bbox_envelope_json(),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "resultOffset": offset,
        "resultRecordCount": PAGE_SIZE,
        "f": "geojson",
    }
    resp = requests.get(FEATURE_SERVER_QUERY_URL, params=params, timeout=120)
    resp.raise_for_status()
    return gpd.read_file(resp.text)


def fetch_buildings():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Fetching building footprints for: {STUDY_AREA_NAME}")
    print(f"BBox (west, south, east, north): {BBOX_LEFT_BOTTOM_RIGHT_TOP}")

    total = _get_count()
    print(f"Server reports {total} buildings intersecting bbox "
          f"(will page in batches of {PAGE_SIZE})")

    pages = []
    offset = 0
    while offset < total:
        print(f"  fetching records {offset}-{min(offset + PAGE_SIZE, total)} "
              f"of {total}...")
        gdf_page = _fetch_page(offset)
        if len(gdf_page) == 0:
            break
        pages.append(gdf_page)
        offset += PAGE_SIZE

    gdf = gpd.GeoDataFrame(pd.concat(pages, ignore_index=True), crs="EPSG:4326")
    print(f"Downloaded {len(gdf)} raw building features")

    # Clip precisely to the bounding box (server-side filter is intersects,
    # which can include geometry that extends slightly past the box).
    bbox_geom = box(WEST, SOUTH, EAST, NORTH)
    gdf_clipped = gpd.clip(gdf, bbox_geom)
    print(f"Clipped to {len(gdf_clipped)} building features within bbox")

    gdf_clipped.to_file(BUILDINGS_PATH, driver="GeoJSON")
    print(f"Saved buildings to: {BUILDINGS_PATH}")

    print("\n--- Sanity check stats ---")
    print(f"Columns: {list(gdf_clipped.columns)}")
    print(f"CRS: {gdf_clipped.crs}")
    print(f"Bounds: {gdf_clipped.total_bounds}")
    if "HEIGHT" in gdf_clipped.columns:
        heights = gdf_clipped["HEIGHT"].dropna()
        print(f"HEIGHT stats (ft): min={heights.min():.1f}, "
              f"max={heights.max():.1f}, mean={heights.mean():.1f}, "
              f"non-null={len(heights)}/{len(gdf_clipped)}")

    return gdf_clipped


if __name__ == "__main__":
    fetch_buildings()
