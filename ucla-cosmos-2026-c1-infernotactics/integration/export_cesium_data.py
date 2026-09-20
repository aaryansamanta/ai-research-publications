"""Export Palisades-focused browser data from the project assets."""

import json
import os
import sys

import networkx as nx
from shapely.geometry import box, mapping, shape
from shapely.ops import transform
from pyproj import Transformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_ROOT, "infernotactics", "src")
sys.path.insert(0, SRC)

from data_pipeline.config import DATA_DIR, REAL_DEPOTS_PATH, ROADS_GRAPHML_PATH  # noqa: E402

DISPLAY_BBOX = {"north": 34.105, "south": 34.030, "east": -118.485, "west": -118.605}
OUT_DIR = os.path.join(PROJECT_ROOT, "integration", "static", "data")


def export_buildings():
    source = os.path.join(DATA_DIR, "buildings.geojson")
    with open(source, encoding="utf-8") as f:
        data = json.load(f)
    bounds = box(DISPLAY_BBOX["west"], DISPLAY_BBOX["south"], DISPLAY_BBOX["east"], DISPLAY_BBOX["north"])
    features = []
    for feature in data["features"]:
        geom = shape(feature["geometry"])
        clipped = geom.intersection(bounds)
        if clipped.is_empty:
            continue
        props = feature.get("properties", {})
        features.append({
            "type": "Feature",
            "geometry": mapping(clipped.simplify(0.00001, preserve_topology=True)),
            "properties": {
                "id": props.get("OBJECTID", len(features)),
                "height_m": min(float(props.get("HEIGHT") or 0.0) * 0.3048, 150.0),
                "status": "intact",
            },
        })
    with open(os.path.join(OUT_DIR, "palisades_buildings.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"buildings: {len(features)}")


def export_roads():
    graph = nx.read_graphml(ROADS_GRAPHML_PATH)
    features = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        try:
            x1, y1 = float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])
            x2, y2 = float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (DISPLAY_BBOX["west"] <= x1 <= DISPLAY_BBOX["east"] or DISPLAY_BBOX["west"] <= x2 <= DISPLAY_BBOX["east"]):
            continue
        if not (DISPLAY_BBOX["south"] <= y1 <= DISPLAY_BBOX["north"] or DISPLAY_BBOX["south"] <= y2 <= DISPLAY_BBOX["north"]):
            continue
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0] if highway else "unclassified"
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[x1, y1], [x2, y2]]},
            "properties": {"id": f"{u}-{v}-{key}", "highway": str(highway)},
        })
    with open(os.path.join(OUT_DIR, "palisades_roads.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"roads: {len(features)}")


def export_depots():
    with open(REAL_DEPOTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    stations = []
    for station in data["stations"]:
        inside = DISPLAY_BBOX["west"] <= station["lon"] <= DISPLAY_BBOX["east"] and DISPLAY_BBOX["south"] <= station["lat"] <= DISPLAY_BBOX["north"]
        if inside or station["travel_mode"] == "air_straight_line":
            stations.append({
                "station_id": station["station_id"], "name": station["station_name"],
                "lat": station["lat"], "lon": station["lon"],
                "travel_mode": station["travel_mode"], "roster": station["roster"],
            })
    with open(os.path.join(OUT_DIR, "palisades_depots.json"), "w", encoding="utf-8") as f:
        json.dump(stations, f, indent=2)


def export_config():
    with open(os.path.join(OUT_DIR, "display_config.json"), "w", encoding="utf-8") as f:
        json.dump({"display_bbox": DISPLAY_BBOX, "center": {"lat": 34.0725, "lon": -118.5425}}, f, indent=2)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    export_buildings()
    export_roads()
    export_depots()
    export_config()
