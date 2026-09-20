"""
Pull the real Palisades Fire perimeter for numerical validation of
fire_sim.py's simulated spread against actual burned extent.

Source: LA County Public Works' "Palisades and Eaton Dissolved Fire
Perimeters (2025)" dataset (LA GeoHub), a public, unauthenticated
FeatureServer:
  https://services.arcgis.com/RmCCgQtiZLDCtblq/arcgis/rest/services/Palisades_and_Eaton_Dissolved_Fire_Perimeters_as_of_20250121/FeatureServer
Layer 1 ("Palisades_Perimeter_20250121") holds 21 disjoint "Heat Perimeter"
polygons -- the main fire body plus several detached spot-fire/outlier
pieces -- all dated to a single snapshot, Jan 21 2025 (~13.5 days after the
real ignition). This is a "dissolved" product (per the dataset's own
description: NIFC FIRIS daily perimeters merged into one shape) -- i.e. this
is the ONE perimeter snapshot publicly available through this route, NOT a
timestamped daily progression. Searched (WFIGS current-year layers, the
multi-decade WFIGS_Interagency_Perimeters history layer, CA_Perimeters_NIFC_
FIRIS_public_view) turned up nothing with sub-14-day granularity for this
specific fire -- WFIGS's "current year" perimeter layers roll over/clear
after a fire is closed out and the older multi-year history layer's own
documentation states it only covers "thru the 2024 fire season" (this fire
is Jan 2025). See wfigs_perimeter_validation.py for how this single snapshot
is actually used given that constraint.
"""

import os
import sys

import geopandas as gpd
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import DATA_DIR  # noqa: E402

FEATURE_SERVER_QUERY_URL = (
    "https://services.arcgis.com/RmCCgQtiZLDCtblq/arcgis/rest/services/"
    "Palisades_and_Eaton_Dissolved_Fire_Perimeters_as_of_20250121/FeatureServer/1/query"
)
PERIMETER_PATH = os.path.join(DATA_DIR, "palisades_perimeter_20250121.geojson")


def fetch_perimeter():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Fetching real Palisades Fire perimeter (as of 2025-01-21) from: {FEATURE_SERVER_QUERY_URL}")

    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
    }
    resp = requests.get(FEATURE_SERVER_QUERY_URL, params=params, timeout=120)
    resp.raise_for_status()
    gdf = gpd.read_file(resp.text)
    # f=geojson responses from ArcGIS are always WGS84 (EPSG:4326) regardless
    # of the layer's storage SR -- geopandas reads this correctly from the
    # response itself. Fallback only in case a future response omits it.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    print(f"Downloaded {len(gdf)} perimeter polygon features")
    print(f"Columns: {list(gdf.columns)}")
    print(f"CRS: {gdf.crs}")

    gdf.to_file(PERIMETER_PATH, driver="GeoJSON")
    print(f"Saved to: {PERIMETER_PATH}")

    total_area_m2 = gdf.to_crs("EPSG:5070").geometry.area.sum()
    total_acres = total_area_m2 / 4046.8564224
    print(f"\nSanity check: total dissolved perimeter area = {total_area_m2:,.0f} sq m "
          f"({total_acres:,.1f} acres). Real Palisades Fire's documented final size is "
          f"~23,400 acres -- this Jan 21 snapshot is ~13.5 days post-ignition, expected to "
          f"be close to (not necessarily exactly) that final figure.")
    return gdf


if __name__ == "__main__":
    fetch_perimeter()
