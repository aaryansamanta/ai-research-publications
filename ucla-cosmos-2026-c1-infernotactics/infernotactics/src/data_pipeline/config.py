"""
Study area configuration for InfernoTactics AI.

Covers the Palisades Fire footprint and surrounding area: Topanga State Park
and the Pacific Palisades hills, south through Santa Monica Canyon, east
through Brentwood / Bel-Air, to Westwood / UCLA.

Coordinates are approximate real-world landmarks used to build the bounding
box (WGS84, EPSG:4326):
  - Topanga State Park (Trippet Ranch):  34.0958 N, -118.5844 W
  - Pacific Palisades Village:            34.0473 N, -118.5265 W
  - Santa Monica Canyon / PCH:            34.0367 N, -118.5233 W
  - Getty Center:                         34.0780 N, -118.4741 W
  - UCLA (Royce Hall):                    34.0722 N, -118.4426 W
  - Westwood Village:                     34.0611 N, -118.4462 W
"""

# Bounding box: (north, south, east, west) in decimal degrees, WGS84.
NORTH = 34.150   # above Topanga State Park's northern ridgeline, near Mulholland Dr
SOUTH = 34.030   # Santa Monica Canyon / PCH, south of the Palisades bluffs
EAST = -118.440  # east of UCLA campus / Westwood Village
WEST = -118.605  # west of Topanga State Park, near Topanga Canyon Blvd & PCH

BBOX_NORTH_SOUTH_EAST_WEST = (NORTH, SOUTH, EAST, WEST)

# (left, bottom, right, top) = (west, south, east, north) -- the order used
# by osmnx>=2.0, geopandas .cx[], and most GIS bbox conventions.
BBOX_LEFT_BOTTOM_RIGHT_TOP = (WEST, SOUTH, EAST, NORTH)

STUDY_AREA_NAME = "Palisades Fire Area, Los Angeles, CA"

# Local data paths
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

ROADS_GRAPHML_PATH = os.path.join(DATA_DIR, "roads.graphml")
BUILDINGS_PATH = os.path.join(DATA_DIR, "buildings.geojson")
ELEVATION_PATH = os.path.join(DATA_DIR, "elevation.tif")
POPULATION_PATH = os.path.join(DATA_DIR, "population.tif")

# Real LAFD depot locations (one per RL resource type) -- see the file itself
# for sourcing notes. Lives alongside this config rather than under DATA_DIR
# since it's small, hand-curated, and version-controlled, not a pipeline output.
REAL_DEPOTS_PATH = os.path.join(_THIS_DIR, "real_depots.json")

# Real Jan 7-8 2025 wind/humidity time series (NOAA ASOS, Santa Monica Airport
# station KSMO) -- see fetch_weather.py for sourcing/cleaning details.
WEATHER_STATION = "SMO"
WEATHER_CSV_PATH = os.path.join(DATA_DIR, "palisades_weather_jan2025.csv")
