"""
Pull the real drivable road network for the Palisades Fire study area from
OpenStreetMap via OSMnx, and save it locally for use by the simulation env.
"""

import os
import sys

import osmnx as ox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import (
    BBOX_LEFT_BOTTOM_RIGHT_TOP,
    DATA_DIR,
    ROADS_GRAPHML_PATH,
    STUDY_AREA_NAME,
)


def fetch_roads():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Fetching drivable road network for: {STUDY_AREA_NAME}")
    print(f"BBox (west, south, east, north): {BBOX_LEFT_BOTTOM_RIGHT_TOP}")

    graph = ox.graph_from_bbox(
        bbox=BBOX_LEFT_BOTTOM_RIGHT_TOP,
        network_type="drive",
        simplify=True,
    )

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    print(f"Downloaded graph: {n_nodes} nodes, {n_edges} edges")

    ox.save_graphml(graph, filepath=ROADS_GRAPHML_PATH)
    print(f"Saved graph to: {ROADS_GRAPHML_PATH}")

    # Basic sanity stats
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)
    print("\n--- Sanity check stats ---")
    print(f"Nodes: {len(gdf_nodes)}")
    print(f"Edges: {len(gdf_edges)}")
    print(f"Node CRS: {gdf_nodes.crs}")
    print(f"Node bounds: {gdf_nodes.total_bounds}")
    if "highway" in gdf_edges.columns:
        print("Edge highway-type counts:")
        print(gdf_edges["highway"].astype(str).value_counts().head(10))

    return graph


if __name__ == "__main__":
    fetch_roads()
