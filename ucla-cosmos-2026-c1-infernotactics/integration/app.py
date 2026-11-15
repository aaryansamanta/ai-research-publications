"""
InfernoTactics 3D Simulation — FastAPI backend (v10 rewrite).

Endpoints:
  GET /         -> serve Simulation.html
  GET /world    -> static geometry, zones, coverage, delay (cached)
  GET /simulate -> run full episode, return timeline JSON
  GET /health   -> diagnostic info

Key design:
  - InfernoEnv built ONCE at startup (Dijkstra ~1s), not per request.
  - Episode runs server-side; browser plays the returned timeline over 30s.
  - Fire sent as per-tick diffs, not full grids.
  - Multi-dispatch decode loop (canonical, from eval_relative.py).
  - Response-delay gate: AI held for N ticks derived from zone coverage.
"""

import math
import os
import sys
import requests
import time
from datetime import timezone

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: import from infernotactics/src, NOT best_model/src
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "infernotactics", "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from env.inferno_env import (  # noqa: E402
    InfernoEnv,
    TRAINING_IGNITION_POINT,
    RESOURCE_TYPES,
    SCALAR_KEYS,
    ZONE_SIZE_CELLS,
    TICK_DURATION_MINUTES,
    MAX_TICKS,
    FIRE_START_UTC,
    GROUND_RESOURCE_TYPES,
    RESOURCE_COUNTS,
    RESOURCE_DELAY_CONFIG,
    HELICOPTER_SPEED_MPS,
    flatten_scalars,
)
from env.fire_sim import SAFE, FUEL, THREAT, BLAZE, BURNED_OUT  # noqa: E402
from models.relative_model import RelativeInfernoModel  # noqa: E402
from train.relative_actions import decode_action, TARGET_TYPES, resolve_relative_targets  # noqa: E402
from train.train_relative import MAX_DISPATCH_SLOTS, _forward  # noqa: E402

from fastapi import FastAPI, Query  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pyproj import Transformer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CKPT_PATH = os.path.join(
    REPO_ROOT, "infernotactics", "models",
    "checkpoints_relative_v8", "latest.pt"
)
DEVICE = torch.device("cpu")

# Coverage → delay mapping
MIN_DELAY_TICKS = 1
MAX_DELAY_TICKS = 12  # 24 sim-minutes at 2 min/tick
POST_EPISODE_TICKS = 14  # aftermath frames after containment

# Grid corner coordinates (EPSG:5070 → WGS84, measured)
GRID_CORNERS = {
    "nw": [34.11241, -118.62922],
    "ne": [34.14986, -118.44000],
    "sw": [34.03024, -118.60490],
    "se": [34.06766, -118.41587],
}

# ---------------------------------------------------------------------------
# Globals — initialized once at startup
# ---------------------------------------------------------------------------
env: InfernoEnv = None
model: RelativeInfernoModel = None
transformer_to_wgs84: Transformer = None
transformer_from_wgs84: Transformer = None
world_cache: dict = None

app = FastAPI(title="InfernoTactics 3D Fire Simulator — v10")
app.mount("/static", StaticFiles(directory=CURRENT_DIR), name="static")


# ===========================================================================
# Coordinate & Camera helpers
# ===========================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_destination_point(lat, lon, bearing_deg, distance_km):
    R = 6371.0
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)

    dest_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(distance_km / R) +
        math.cos(lat_rad) * math.sin(distance_km / R) * math.cos(bearing_rad)
    )
    dest_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat_rad),
        math.cos(distance_km / R) - math.sin(lat_rad) * math.sin(dest_lat_rad)
    )
    return [math.degrees(dest_lat_rad), math.degrees(dest_lon_rad)]

def fetch_camera_polygons():
    """Fetch cameras from ALERTWest API and return a list of shapely Polygons."""
    from shapely.geometry import Polygon
    url = "https://api.cdn.prod.alertwest.com/api/firecams/v0/cameras"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    
    CENTER_LAT, CENTER_LON = 34.0500, -118.5250
    NEARBY_RADIUS_KM = 50.0
    
    try:
        resp = requests.get(url, headers=headers, timeout=2)
        resp.raise_for_status()
        raw_cameras = resp.json()
    except Exception as e:
        print(f"[warning] Failed to fetch cameras for coverage: {e}")
        return []

    cam_polys = []
    for cam in raw_cameras:
        try:
            site = cam.get("site", {})
            lat_val = site.get("latitude") or cam.get("latitude")
            lon_val = site.get("longitude") or cam.get("longitude")
            if lat_val is None or lon_val is None:
                continue

            lat, lon = float(lat_val), float(lon_val)
            dist = haversine(CENTER_LAT, CENTER_LON, lat, lon)
            if dist > NEARBY_RADIUS_KM:
                continue

            pos = cam.get("position", {})
            pan_val = pos.get("pan") if isinstance(pos, dict) else cam.get("pan")
            pan = float(pan_val) if pan_val is not None else 0.0
            
            fov_deg = 60.0
            range_km = 18.0 if dist <= 12.0 else dist + 15.0

            start_angle = pan - (fov_deg / 2.0)
            end_angle = pan + (fov_deg / 2.0)

            sector_points = [[lat, lon]]
            num_steps = 16
            for i in range(num_steps + 1):
                angle = start_angle + (end_angle - start_angle) * (i / num_steps)
                pt = calculate_destination_point(lat, lon, angle, range_km)
                sector_points.append(pt)
            sector_points.append([lat, lon])

            # Polygon uses (lon, lat)
            poly = Polygon([(pt[1], pt[0]) for pt in sector_points])
            cam_polys.append(poly)
        except Exception:
            continue
            
    print(f"[startup] Built {len(cam_polys)} camera FOV polygons.")
    return cam_polys

def grid_to_epsg5070(row, col, meta):
    """Convert grid (row, col) → EPSG:5070 (x, y) using the affine transform."""
    a, b, c, d, e, f = meta["transform"]
    x = a * (col + 0.5) + b * (row + 0.5) + c
    y = d * (col + 0.5) + e * (row + 0.5) + f
    return x, y


def epsg5070_to_latlon(x, y):
    """EPSG:5070 (x, y) → WGS84 (lat, lon)."""
    lon, lat = transformer_to_wgs84.transform(x, y)
    return float(lat), float(lon)

def latlon_to_epsg5070(lat, lon):
    x, y = transformer_from_wgs84.transform(lon, lat)
    return float(x), float(y)

def epsg5070_to_grid(x, y, meta):
    a, b, c, d, e, f = meta["transform"]
    x_prime, y_prime = x - c, y - f
    det = a * e - b * d
    if det == 0: return 0, 0
    col = int((e * x_prime - b * y_prime) / det - 0.5)
    row = int((a * y_prime - d * x_prime) / det - 0.5)
    return row, col

def grid_to_latlon(row, col, meta):
    """Grid (row, col) → WGS84 (lat, lon), exact via pyproj."""
    x, y = grid_to_epsg5070(row, col, meta)
    return epsg5070_to_latlon(x, y)

def latlon_to_grid(lat, lon, meta):
    """WGS84 (lat, lon) → Grid (row, col) exact inverse."""
    x, y = latlon_to_epsg5070(lat, lon)
    row, col = epsg5070_to_grid(x, y, meta)
    row = max(0, min(meta["height"] - 1, row))
    col = max(0, min(meta["width"] - 1, col))
    return row, col


# ===========================================================================
# Coverage / delay computation
# ===========================================================================

def compute_zone_data(env):
    """Compute coverage, delay, polygon corners, and metadata for all zones."""
    meta = env.meta

    # Best arrival time across all resource types per zone
    best_arrival = {}
    for z in range(env.n_zones):
        best_arrival[z] = min(
            env.zone_travel_time_s[t][z] for t in RESOURCE_TYPES
        )

    finite_arrivals = [v for v in best_arrival.values() if math.isfinite(v)]
    lo = min(finite_arrivals) if finite_arrivals else 0.0
    hi = max(finite_arrivals) if finite_arrivals else 1.0
    span = max(hi - lo, 1e-6)

    print("[startup] Fetching ALERTCalifornia cameras for coverage...")
    cam_polys = fetch_camera_polygons()
    from shapely.geometry import Polygon

    zones_data = []
    for zone in env.zones:
        zid = zone["zone_id"]
        r0, r1 = zone["row_range"]
        c0, c1 = zone["col_range"]

        # 4-corner polygon (exact pyproj conversion)
        corners = [
            grid_to_latlon(r0, c0, meta),  # top-left
            grid_to_latlon(r0, c1, meta),  # top-right
            grid_to_latlon(r1, c1, meta),  # bottom-right
            grid_to_latlon(r1, c0, meta),  # bottom-left
        ]

        # Centroid
        centroid = grid_to_latlon(zone["centroid_row"], zone["centroid_col"], meta)

        ba = best_arrival[zid]
        if not math.isfinite(ba):
            ba = hi  # treat inf as worst

        # Coverage based on ALERTCalifornia camera intersections
        poly_shapely = Polygon([(c[1], c[0]) for c in corners])
        cov_count = sum(1 for cam_poly in cam_polys if cam_poly.intersects(poly_shapely))
        # Mapping rules from Response_delay script (shifted scale)
        if cov_count >= 14:
            cov = 1.0      # Purple
        elif cov_count == 13:
            cov = 0.9      # Blue
        elif cov_count == 12:
            cov = 0.8      # Green
        elif cov_count == 11:
            cov = 0.7      # Yellow
        elif cov_count == 10:
            cov = 0.4      # Orange
        else:
            cov = 0.15     # Red

        # Delay (realistic: worse coverage → longer hold)
        delay = round(MIN_DELAY_TICKS + (1.0 - cov) * (MAX_DELAY_TICKS - MIN_DELAY_TICKS))

        # Ground reachable?
        ground_reachable = any(
            math.isfinite(env.zone_travel_time_s[t][zid])
            for t in GROUND_RESOURCE_TYPES
        )

        # Per-type travel times
        per_type = {}
        for t in RESOURCE_TYPES:
            tt = env.zone_travel_time_s[t][zid]
            per_type[t] = round(tt, 1) if math.isfinite(tt) else None

        zones_data.append({
            "id": zid,
            "row": zid // 8,
            "col": zid % 8,
            "polygon": [[lat, lon] for lat, lon in corners],
            "centroid": list(centroid),
            "best_arrival_s": round(ba, 1),
            "coverage": cov,
            "delay_ticks": delay,
            "cam_count": cov_count,
            "ground_reachable": ba < 1e5,
            "building_cells": zone["building_cells"],
            "is_water": zone["is_water"],
            "travel_times": per_type,
        })

    return zones_data, lo, hi


def compute_station_data(env):
    """Convert station information to lat/lon with rosters."""
    stations = []
    for station in env.stations:
        stations.append({
            "id": str(station["station_id"]),
            "name": station["station_name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "mode": station["travel_mode"],
            "roster": dict(station["roster"]),
        })
    return stations


# ===========================================================================
# Unit tracker — position reconstruction
# ===========================================================================

class UnitTracker:
    """Track positions of all 15 resource units across ticks.

    Reconstructs lat/lon positions from unit lifecycle state,
    interpolating ground units along their route polyline and
    helicopters along a straight-line path with altitude profile.
    """

    # Unit state mapping for JSON
    STATE_MAP = {
        "available": "idle",
        "preparing": "prep",
        "traveling": "move",
        "arrival_setup": "setup",
        "deployed": "work",
    }

    def __init__(self, env):
        self.env = env
        self.meta = env.meta
        # Pre-compute station lat/lon
        self.station_latlon = {}
        for station in env.stations:
            self.station_latlon[str(station["station_id"])] = (
                station["lat"], station["lon"]
            )
        # Unit IDs (e.g., "helicopter:0", "water_team:1")
        self.unit_ids = {}
        for rtype in RESOURCE_TYPES:
            for i in range(len(env.resources[rtype])):
                uid = f"{rtype}:{i}"
                self.unit_ids[(rtype, i)] = uid

    def _get_target_latlon(self, env, zone_idx):
        import numpy as np
        zone = env.zones[zone_idx]
        r0, r1 = zone["row_range"]
        c0, c1 = zone["col_range"]
        
        region = env.sim.state[r0:r1, c0:c1]
        active = np.argwhere((region == THREAT) | (region == BLAZE))
        
        # If no active fire in this sector, check for burned out fire so they stay where the fire was
        if len(active) == 0:
            active = np.argwhere((region == BURNED_OUT))
            
        if len(active) > 0:
            fire_r = r0 + active[:, 0].mean()
            fire_c = c0 + active[:, 1].mean()
            return grid_to_latlon(fire_r, fire_c, self.meta)
        else:
            # Preventative dispatch: find the nearest active fire anywhere on the map
            global_active = np.argwhere((env.sim.state == THREAT) | (env.sim.state == BLAZE))
            if len(global_active) > 0:
                # Find the nearest one to the centroid of the assigned zone
                cr, cc = zone["centroid_row"], zone["centroid_col"]
                dists = (global_active[:, 0] - cr)**2 + (global_active[:, 1] - cc)**2
                nearest_idx = np.argmin(dists)
                fire_r, fire_c = global_active[nearest_idx]
                return grid_to_latlon(fire_r, fire_c, self.meta)
            else:
                return grid_to_latlon(zone["centroid_row"], zone["centroid_col"], self.meta)

    def snapshot(self, env):
        """Return unit positions for the current tick."""
        units = []
        for rtype in RESOURCE_TYPES:
            for i, unit in enumerate(env.resources[rtype]):
                uid = self.unit_ids[(rtype, i)]
                state = self.STATE_MAP.get(unit["state"], "idle")
                sid = str(unit["station_id"])
                station_lat, station_lon = self.station_latlon[sid]

                if state == "idle":
                    lat, lon = station_lat, station_lon
                    alt = 0 if rtype != "helicopter" else 50
                elif state in ("prep",):
                    lat, lon = station_lat, station_lon
                    alt = 0 if rtype != "helicopter" else 100
                elif state == "move":
                    # Interpolate along route
                    if unit["target_zone"] is not None:
                        target_lat, target_lon = self._get_target_latlon(env, unit["target_zone"])
                        # Progress fraction
                        total_ticks = unit.get("pending_travel_ticks", 1) or 1
                        remaining = unit["remaining_ticks"]
                        progress = 1.0 - (remaining / total_ticks)
                        progress = max(0.0, min(1.0, progress))

                        lat = station_lat + (target_lat - station_lat) * progress
                        lon = station_lon + (target_lon - station_lon) * progress

                        if rtype == "helicopter":
                            # Altitude arc: rise to cruise, then descend
                            cruise_alt = 420  # meters
                            alt = cruise_alt * math.sin(math.pi * progress) if progress < 1 else 50
                        else:
                            alt = 0
                    else:
                        lat, lon = station_lat, station_lon
                        alt = 0
                elif state in ("setup", "work"):
                    # At target zone
                    if unit["target_zone"] is not None:
                        target_lat, target_lon = self._get_target_latlon(env, unit["target_zone"])
                        lat, lon = target_lat, target_lon
                        alt = 80 if rtype == "helicopter" else 0
                    else:
                        lat, lon = station_lat, station_lon
                        alt = 0
                else:
                    lat, lon = station_lat, station_lon
                    alt = 0

                unit_data = {
                    "u": uid,
                    "s": state,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "alt": round(alt, 1),
                }
                if unit["target_zone"] is not None:
                    unit_data["z"] = unit["target_zone"]
                    # Progress
                    if state == "move" and unit.get("pending_travel_ticks"):
                        total = unit["pending_travel_ticks"]
                        remaining = unit["remaining_ticks"]
                        unit_data["p"] = round(1.0 - remaining / total, 2)

                units.append(unit_data)

        return units


# ===========================================================================
# Episode rollout
# ===========================================================================

def run_episode(env, model, delay_ticks, ig_row=None, ig_col=None, seed=9100, scenario="anchor"):
    """Run a complete episode with response-delay gate.

    Returns the timeline dict matching the spec's §5.2 contract.
    """
    ig_point = TRAINING_IGNITION_POINT if ig_row is None else (ig_row, ig_col)
    
    obs = env.reset(
        ignition_point=ig_point,
        scenario="single",
        seed=seed,
        use_real_weather=True,
    )

    tracker = UnitTracker(env)

    # Initial fire state
    fire_state = env.sim.state.copy()
    active_indices = np.flatnonzero(np.isin(fire_state.ravel(), [THREAT, BLAZE, BURNED_OUT]))
    initial_fire = [
        [int(idx), int(fire_state.ravel()[idx])]
        for idx in active_indices
    ]

    # Track fire state for diffing
    prev_fire = {}
    for idx in active_indices:
        prev_fire[int(idx)] = int(fire_state.ravel()[idx])

    ticks_data = []
    total_reward = 0.0
    total_destroyed = 0
    done = False
    tick = 0
    post_ticks = 0

    while not done or post_ticks < POST_EPISODE_TICKS:
        is_held = tick < delay_ticks and not done

        # Run policy forward (even during hold, for UI display)
        with torch.no_grad():
            logits, value, _cls, target_zones = _forward(model, obs, env, DEVICE)

        # Determine actions
        available = {r: int(obs["scalars"][f"{r}_available"]) for r in RESOURCE_TYPES}
        policy_actions = []
        actions_to_execute = []

        if scenario != "unmitigated":
            for _ in range(MAX_DISPATCH_SLOTS):
                resource_logits = logits["resource_type"][0].clone()
                mask = torch.tensor(
                    [available[r] > 0 for r in RESOURCE_TYPES],
                    dtype=torch.bool, device=DEVICE
                )
                resource_logits[~mask] = -1e9
                if not bool(mask.any()):
                    break
                ri = int(torch.argmax(resource_logits))
                ti = int(torch.argmax(logits["target"][0, ri]))
                action = decode_action(ri, ti, target_zones)
                if action is None:
                    break

                semantic_target = TARGET_TYPES[ti] if ti < len(TARGET_TYPES) else "unknown"
                policy_actions.append({
                    "r": RESOURCE_TYPES[ri],
                    "t": semantic_target,
                    "z": action[1],
                })

                if not is_held and not done:
                    available[RESOURCE_TYPES[ri]] -= 1
                    actions_to_execute.append(action)

        # Step the environment — ALWAYS with a list
        if done:
            obs, reward, _, info = env.step([])  # aftermath: no actions
            post_ticks += 1
        elif is_held:
            obs, reward, done, info = env.step([])  # held: no actions
        else:
            obs, reward, done, info = env.step(actions_to_execute)  # list!

        if not done or post_ticks == 0:
            total_reward += reward
            total_destroyed += info.get("buildings_destroyed", 0)

        # Fire diff
        new_fire = env.sim.state.ravel()
        fire_add = []
        fire_del = []

        # Find new/changed fire cells
        active_mask = np.isin(new_fire, [THREAT, BLAZE, BURNED_OUT])
        new_active = np.flatnonzero(active_mask)

        for idx in new_active:
            idx_int = int(idx)
            state_val = int(new_fire[idx_int])
            if idx_int not in prev_fire or prev_fire[idx_int] != state_val:
                fire_add.append([idx_int, state_val])
                prev_fire[idx_int] = state_val

        # Find cells that returned to safe/fuel
        to_remove = []
        for idx_int, old_state in prev_fire.items():
            new_state = int(new_fire[idx_int])
            if new_state in (SAFE, FUEL):
                fire_del.append(idx_int)
                to_remove.append(idx_int)
        for idx_int in to_remove:
            del prev_fire[idx_int]

        # Weather
        weather = info.get("weather", {})
        wind_speed = weather.get("wind_speed_mph", 0)
        wind_dir = weather.get("wind_direction_deg", 0)
        humidity = weather.get("humidity_pct", 0)

        # Sim time
        elapsed_minutes = tick * TICK_DURATION_MINUTES
        sim_dt = FIRE_START_UTC.timestamp() + elapsed_minutes * 60
        # PST = UTC - 8
        pst_hour = int((sim_dt / 3600) % 24) - 8
        if pst_hour < 0:
            pst_hour += 24
        pst_min = int((sim_dt / 60) % 60)
        sim_time = f"{pst_hour:02d}:{pst_min:02d}"

        # State counts
        counts = info.get("state_counts", {})
        active_count = counts.get("Threat", 0) + counts.get("Blaze", 0)
        blaze_count = counts.get("Blaze", 0)
        burned_count = counts.get("Burned Out", 0)

        # Ready counts (snapshot at decision time)
        ready = {r: int(obs["scalars"][f"{r}_available"]) for r in RESOURCE_TYPES}

        # Unit positions
        units = tracker.snapshot(env)

        # Dispatch events
        dispatch_events = []
        for d in info.get("dispatch", []):
            dispatch_events.append({
                "r": d["resource_type"],
                "z": d["target_zone"],
                "st": d["status"],
                "eta": d.get("eta_ticks"),
                "travel_s": round(d.get("travel_time_s", 0), 1),
                "station": str(d.get("station_id", "")),
            })

        # Effect events
        effect_events = []
        for ev in info.get("resource_events", []):
            zone = env.zones[ev["zone"]]
            lat, lon = grid_to_latlon(ev["row"], ev["col"], env.meta)
            effect_events.append({
                "r": ev["resource_type"],
                "z": ev["zone"],
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "n": ev["cells_affected"],
                "ok": ev["success"],
            })

        # Building losses
        losses = []
        for bl in info.get("building_destruction_events", []):
            lat, lon = grid_to_latlon(bl["row"], bl["col"], env.meta)
            losses.append({
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "mult": round(bl["multiplier"], 2),
                "evac": bl["evacuated"],
            })

        tick_data = {
            "t": tick,
            "held": is_held,
            "post": done and post_ticks > 0,
            "sim_time": sim_time,
            "wind": round(wind_speed, 1),
            "dir": round(wind_dir, 1),
            "rh": round(humidity, 1),
            "reward": round(reward, 2),
            "cum": round(total_reward, 2),
            "value": round(float(value.item()), 3),
            "active": active_count,
            "blaze": blaze_count,
            "burned": burned_count,
            "destroyed": info.get("buildings_destroyed", 0),
            "destroyed_total": total_destroyed,
            "fire": {"add": fire_add, "del": fire_del},
            "ready": ready,
            "units": units,
            "actions": policy_actions,
            "dispatch": dispatch_events,
            "effects": effect_events,
            "losses": losses,
        }

        ticks_data.append(tick_data)
        tick += 1

        # Safety: absolute cap allowing complete burn-out or containment
        if tick > 1000:
            break

    # Compute ignition lat/lon
    ig_row, ig_col = ig_point
    ig_lat, ig_lon = grid_to_latlon(ig_row, ig_col, env.meta)
    ig_r = min(3, max(0, int(ig_row * 4 / env.height)))
    ig_c = min(7, max(0, int(ig_col * 8 / env.width)))
    ig_zone = ig_r * 8 + ig_c

    # Get zone data for delay explanation
    zone_info = world_cache["zones"][ig_zone] if world_cache else {}

    result = {
        "scenario": {
            "key": "anchor",
            "label": "Skull Rock trailhead",
            "ignition": {
                "row": ig_row, "col": ig_col,
                "lat": round(ig_lat, 6), "lon": round(ig_lon, 6),
                "zone": ig_zone,
            },
            "delay": {
                "ticks": delay_ticks,
                "zone": ig_zone,
                "coverage": zone_info.get("coverage", 0.15),
                "best_arrival_s": zone_info.get("best_arrival_s", 280.0),
                "reason": (
                    f"Sector {ig_zone} has no road access; nearest air response "
                    f"{zone_info.get('best_arrival_s', 280.0):.0f}s. "
                    f"AI held {delay_ticks} ticks."
                ),
            },
        },
        "outcome": {
            "ticks": tick,
            "reward": round(total_reward, 2),
            "destroyed": total_destroyed,
            "contained": info.get("contained", False),
            "timeout": info.get("timeout", False),
            "frames": len(ticks_data),
        },
        "initial_fire": initial_fire,
        "ticks": ticks_data,
    }

    return result


# ===========================================================================
# FastAPI endpoints
# ===========================================================================

@app.on_event("startup")
async def startup():
    global env, model, transformer_to_wgs84, transformer_from_wgs84, world_cache

    print("[startup] Initializing InfernoEnv (Dijkstra routing)...")
    t0 = time.time()
    env = InfernoEnv(seed=9100)
    print(f"[startup] InfernoEnv ready in {time.time() - t0:.1f}s")

    # Coordinate transformer
    transformer_to_wgs84 = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    transformer_from_wgs84 = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

    # Load model
    print(f"[startup] Loading checkpoint: {CKPT_PATH}")
    obs = env.reset(seed=9100)
    model = RelativeInfernoModel(
        n_grid_channels=obs["grid"].shape[0],
        n_scalars=len(SCALAR_KEYS),
        n_resources=len(RESOURCE_TYPES),
        n_zones=env.n_zones,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    print("[startup] Model loaded, strict=True passed")

    # Build world cache
    print("[startup] Computing zone geometry, coverage, delays...")
    zones_data, travel_lo, travel_hi = compute_zone_data(env)
    stations_data = compute_station_data(env)

    # Pre-compute the 50x50 pixel heatmap grid for the UI
    print("[startup] Computing dense pixel coverage grid (50x50)...")
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    
    ROWS, COLS = 50, 50
    # Generate 51x51 corner points that perfectly follow the tilted simulation grid
    grid_pts = []
    for r in range(ROWS + 1):
        row_pts = []
        env_r = min(int(r * env.height / ROWS), env.height - 1)
        if r == ROWS: env_r = env.height - 1
        
        for c in range(COLS + 1):
            env_c = min(int(c * env.width / COLS), env.width - 1)
            if c == COLS: env_c = env.width - 1
            
            lat, lon = grid_to_latlon(env_r, env_c, env.meta)
            row_pts.append((lon, lat)) # shapely uses (x, y) = (lon, lat)
        grid_pts.append(row_pts)
    
    # Re-fetch cameras to build tree for this operation
    try:
        cam_polys = fetch_camera_polygons()
        tree = STRtree(cam_polys)
    except Exception as e:
        print(f"Error fetching cameras for dense grid: {e}")
        cam_polys = []
        tree = None

    pixel_grid = []
    if tree is not None and len(cam_polys) > 0:
        cells = [
            Polygon([grid_pts[r][c], grid_pts[r][c+1], grid_pts[r+1][c+1], grid_pts[r+1][c]])
            for r in range(ROWS) for c in range(COLS)
        ]
        c_idx, m_idx = tree.query(cells, predicate="intersects")
        counts_arr = np.bincount(c_idx, minlength=len(cells))
        pixel_grid = counts_arr.reshape(ROWS, COLS).tolist()
    else:
        pixel_grid = [[0] * COLS for _ in range(ROWS)]
    
    world_cache = {
        "grid": {
            "width": env.width,
            "height": env.height,
            "cell_size_m": float(env.meta["cell_size_m"]),
            "zone_rows": 4,
            "zone_cols": 8,
            "zone_size_cells": ZONE_SIZE_CELLS,
            "corners": GRID_CORNERS,
            "transform": env.meta.get("transform"),
        },
        "zones": zones_data,
        "stations": stations_data,
        "meta": {
            "checkpoint": os.path.basename(os.path.dirname(CKPT_PATH)) + "/latest.pt",
            "traffic_mode": env.traffic_mode,
            "tick_minutes": TICK_DURATION_MINUTES,
            "max_ticks": MAX_TICKS,
            "fire_start_utc": FIRE_START_UTC.isoformat(),
            "states": {"safe": SAFE, "fuel": FUEL, "threat": THREAT, "blaze": BLAZE, "burned": BURNED_OUT},
            "resource_types": list(RESOURCE_TYPES),
            "roster": dict(RESOURCE_COUNTS),
        },
        "pixel_coverage": {
            "rows": ROWS,
            "cols": COLS,
            "points": grid_pts,
            "grid": pixel_grid
        }
    }

    print(f"[startup] Processed {len(zones_data)} zones (8x4 grid)")

    print("[startup] Ready.")


SIMULATION_HTML_PATH = os.path.join(CURRENT_DIR, "Simulation.html")
@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open(SIMULATION_HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })


@app.get("/world")
async def get_world():
    """Static geometry, zones, stations — called once on page load."""
    return JSONResponse(world_cache)


@app.get("/simulate")
async def simulate(
    scenario: str = Query("anchor"),
    seed: int = Query(9100),
    delay: str = Query("auto"),
    ig_lat: float = Query(None),
    ig_lon: float = Query(None)
):
    """Run the full episode server-side and return the timeline JSON."""
    t0 = time.time()

    ig_row, ig_col = TRAINING_IGNITION_POINT
    if ig_lat is not None and ig_lon is not None:
        ig_row, ig_col = latlon_to_grid(ig_lat, ig_lon, world_cache["grid"])

    # Determine delay based on dynamic ignition zone
    ig_r = min(3, max(0, int(ig_row * 4 / env.height)))
    ig_c = min(7, max(0, int(ig_col * 8 / env.width)))
    ig_zone = ig_r * 8 + ig_c

    if delay == "auto":
        try:
            zone_data = next(z for z in world_cache["zones"] if z["id"] == ig_zone)
            delay_ticks = zone_data["delay_ticks"]
        except StopIteration:
            delay_ticks = 1 # Fallback if out of bounds
    elif delay == "0":
        delay_ticks = 0
    else:
        delay_ticks = int(delay)

    result = run_episode(env, model, delay_ticks, ig_row=ig_row, ig_col=ig_col, seed=seed, scenario=scenario)
    result["_compute_time_s"] = round(time.time() - t0, 2)

    return JSONResponse(result)


@app.get("/health")
async def health():
    """Diagnostic endpoint."""
    return JSONResponse({
        "status": "ok",
        "checkpoint": CKPT_PATH,
        "env_mode": env.traffic_mode if env else None,
        "model_loaded": model is not None,
        "zones": env.n_zones if env else None,
        "grid": f"{env.height}x{env.width}" if env else None,
        "resource_counts": dict(RESOURCE_COUNTS),
        "device": str(DEVICE),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)