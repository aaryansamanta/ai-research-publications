"""
Pull real gridded population data for the Palisades/Brentwood/Bel-Air/Westwood
study area and save it as a small clipped GeoTIFF.

NOTE on source (Census ACS vs. WorldPop): the original plan was Census Bureau
ACS 5-year block-group population (joined to TIGER/Line block-group
polygons). TIGER/Line geometry is freely queryable without a key (see
tigerweb.geo.census.gov's ArcGIS REST service -- confirmed working, 167 block
groups intersect this bbox), but the Census Data API that actually serves the
population ATTRIBUTE (table B01003) now requires a registered API key
(api.census.gov redirects to a "Missing Key" page for keyless requests) --
not something scriptable without a human obtaining a key first.

Fell back to WorldPop's "Global 2000-2020" gridded population count dataset
(2020, USA, unconstrained top-down model, ~1km/30 arcsec resolution),
directly downloadable with no account/key:
https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/USA/usa_ppp_2020_1km_Aggregated.tif
WorldPop also publishes a finer "constrained" 100m dataset for the whole USA,
but that file is ~1.5GB and this environment's bandwidth (~170-190 KB/s
measured) would take ~2.5 hours to pull it -- impractical here, so the 1km
(~50MB) dataset is used instead. This is coarser than block groups would have
been (~1km cells vs. our 30m sim grid -- roughly 33x33 sim cells per
population pixel), a real resolution tradeoff, but it's real 2020 gridded
population data, not a placeholder, and (spot-checked below) already shows
the expected real pattern: near-zero in the Topanga hills, spiking to
5,000-11,700 people/km^2 toward Westwood/UCLA.

Downloads the full national raster to a temp file (deleted after clipping --
not worth keeping ~50MB of nationwide data around for an 18kmx9km study
area), windows out just our bbox (+ a small buffer), and saves that small
clip as POPULATION_PATH, left in its native EPSG:4326 -- grid_builder.py
already reprojects other EPSG:4326 sources (buildings, roads) into the
working EPSG:5070 grid, so this follows that same existing pattern rather
than elevation.tif's (which arrives pre-reprojected from its WMS source).
"""

import os
import sys
import tempfile

import requests
import rasterio
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import (  # noqa: E402
    BBOX_LEFT_BOTTOM_RIGHT_TOP,
    DATA_DIR,
    POPULATION_PATH,
    STUDY_AREA_NAME,
)

WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/USA/"
    "usa_ppp_2020_1km_Aggregated.tif"
)
# Small buffer around the study bbox (degrees) so grid_builder.py's later
# reprojection/resampling never has to extrapolate right at the raster edge.
BBOX_BUFFER_DEG = 0.05


def fetch_population():
    os.makedirs(DATA_DIR, exist_ok=True)
    west, south, east, north = BBOX_LEFT_BOTTOM_RIGHT_TOP
    west, south = west - BBOX_BUFFER_DEG, south - BBOX_BUFFER_DEG
    east, north = east + BBOX_BUFFER_DEG, north + BBOX_BUFFER_DEG

    print(f"Fetching WorldPop 2020 gridded population (~1km) for: {STUDY_AREA_NAME}")
    print(f"BBox + buffer (west, south, east, north): ({west}, {south}, {east}, {north})")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        print(f"Downloading national raster from {WORLDPOP_URL} (~50MB, may take a few minutes)...")
        with requests.get(WORLDPOP_URL, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        print(f"Downloaded {os.path.getsize(tmp_path):,} bytes; clipping to study area...")

        with rasterio.open(tmp_path) as src:
            window = from_bounds(west, south, east, north, src.transform)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            profile = src.profile
            profile.update(
                height=data.shape[0], width=data.shape[1], transform=transform,
                driver="GTiff", compress="lzw",
            )
            with rasterio.open(POPULATION_PATH, "w", **profile) as dst:
                dst.write(data, 1)
    finally:
        os.remove(tmp_path)

    print(f"Saved clipped population raster to: {POPULATION_PATH}")

    print("\n--- Sanity check stats ---")
    nodata = profile.get("nodata")
    valid = data[data != nodata] if nodata is not None else data
    print(f"Shape: {data.shape}, dtype: {data.dtype}, CRS: {profile['crs']}")
    print(f"Population per ~1km^2 cell: min={valid.min():.1f}  max={valid.max():.1f}  "
          f"mean={valid.mean():.1f}  sum={valid.sum():,.0f}")

    return data


if __name__ == "__main__":
    fetch_population()
