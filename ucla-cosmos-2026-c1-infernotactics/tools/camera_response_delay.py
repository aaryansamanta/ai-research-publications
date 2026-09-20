import requests
import pandas as pd
import folium
import math
import json
import numpy as np
from shapely.geometry import Point, Polygon, mapping

# 1. Fetch live data
OFFICIAL_API_URL = "https://api.cdn.prod.alertwest.com/api/firecams/v0/cameras"
headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

try:
    response = requests.get(OFFICIAL_API_URL, headers=headers, timeout=20)
    raw_cameras = response.json()
except Exception as e:
    raw_cameras = []

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_destination_point(lat, lon, bearing_deg, distance_km):
    R = 6371.0
    lat_rad, lon_rad, bearing_rad = map(math.radians, [lat, lon, bearing_deg])
    dest_lat = math.asin(math.sin(lat_rad) * math.cos(distance_km/R) + math.cos(lat_rad) * math.sin(distance_km/R) * math.cos(bearing_rad))
    dest_lon = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_km/R) * math.cos(lat_rad), math.cos(distance_km/R) - math.sin(lat_rad) * math.sin(dest_lat))
    return [math.degrees(dest_lat), math.degrees(dest_lon)]

# Expanded bounds for entire Palisades region
SOUTH, NORTH, WEST, EAST = 34.020, 34.100, -118.620, -118.470
CENTER_LAT, CENTER_LON = 34.0600, -118.5450

palisades_cams = []
for cam in raw_cameras:
    site = cam.get("site", {})
    lat = site.get("latitude") or cam.get("latitude")
    lon = site.get("longitude") or cam.get("longitude")
    if lat is None or lon is None: continue
    lat, lon = float(lat), float(lon)
    dist = haversine(CENTER_LAT, CENTER_LON, lat, lon)
    if dist > 60.0: continue
    pos = cam.get("position", {})
    pan = float(pos.get("pan") or cam.get("pan") or 0.0)
    fov = 60.0
    rng = 18.0 if dist <= 12.0 else dist + 15.0
    pts = [[lat, lon]]
    for i in range(17):
        ang = (pan - fov/2) + (fov * i / 16)
        pts.append(calculate_destination_point(lat, lon, ang, rng))
    pts.append([lat, lon])
    palisades_cams.append({"name": cam.get("name", "Camera"), "lat": lat, "lon": lon, "poly": Polygon([(p[1], p[0]) for p in pts])})

# Detailed grid for coloring
ROWS, COLS = 50, 50
lats = np.linspace(SOUTH, NORTH, ROWS + 1)
lons = np.linspace(WEST, EAST, COLS + 1)

counts = []
cells = []
for r in range(ROWS):
    for c in range(COLS):
        cell_poly = Polygon([(lons[c], lats[r]), (lons[c+1], lats[r]), (lons[c+1], lats[r+1]), (lons[c], lats[r+1])])
        count = sum(1 for cam in palisades_cams if cam["poly"].intersects(cell_poly))
        counts.append(count)
        cells.append(cell_poly)

def get_count_color(c):
    # Shifted color scale per user request
    if c <= 9: return "#ef4444" # Red for 9 or fewer
    if c == 10: return "#f97316" # Orange for 10
    if c == 11: return "#facc15" # Yellow for 11
    if c == 12: return "#22c55e" # Green for 12
    if c == 13: return "#3b82f6" # Blue for 13
    return "#a855f7" # Purple for 14+

features = []
for c, poly in zip(counts, cells):
    features.append({"type": "Feature", "geometry": mapping(poly), "properties": {"cameras": c, "color": get_count_color(c)}})

# Macro Grid (8x4)
MACRO_ROWS, MACRO_COLS = 4, 8
macro_lats = np.linspace(SOUTH, NORTH, MACRO_ROWS + 1)
macro_lons = np.linspace(WEST, EAST, MACRO_COLS + 1)
macro_features = []
for r in range(MACRO_ROWS):
    for c in range(MACRO_COLS):
        m_poly = Polygon([(macro_lons[c], macro_lats[r]), (macro_lons[c+1], macro_lats[r]), (macro_lons[c+1], macro_lats[r+1]), (macro_lons[c], macro_lats[r+1])])
        macro_features.append({"type": "Feature", "geometry": mapping(m_poly)})

m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=12, tiles="CartoDB positron")
folium.GeoJson({"type": "FeatureCollection", "features": features}, style_function=lambda x: {'fillColor': x['properties']['color'], 'color': 'none', 'fillOpacity': 0.5}).add_to(m)
folium.GeoJson({"type": "FeatureCollection", "features": macro_features}, style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 2}).add_to(m)

display(m)