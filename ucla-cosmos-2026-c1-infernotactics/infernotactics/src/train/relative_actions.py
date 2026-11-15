"""Fire-relative action candidates for the v8 generalization experiment.

The historical policy emits absolute macro-zone ids.  This module keeps the
environment's existing ``(resource_type, zone_id)`` API, but exposes a stable
semantic action interface to the learner.  A candidate is resolved from the
current observation on every tick, so ``active_fire`` means the active fire
wherever it is, rather than one memorized geographic zone.
"""

import math

import numpy as np

from env.fire_sim import BLAZE, FUEL, THREAT
from env.inferno_env import BUILDING_PRESENCE_THRESHOLD, LAYER_INDEX, RESOURCE_TYPES


TARGET_TYPES = (
    "active_fire",
    "downwind_fire_front",
    "adjacent_fuel",
    "threatened_population",
    "nearest_reachable_fire",
    "noop",
)
N_TARGET_TYPES = len(TARGET_TYPES)
TARGET_FEATURE_DIM = 10


def _active_mask(fire_state):
    return np.isin(fire_state, (THREAT, BLAZE))


def _eight_connected_dilation(mask):
    result = np.zeros_like(mask, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0 = max(0, dr)
            r1 = min(mask.shape[0], mask.shape[0] + dr)
            c0 = max(0, dc)
            c1 = min(mask.shape[1], mask.shape[1] + dc)
            result[r0:r1, c0:c1] |= mask[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return result


def _downwind_score(rows, cols, fire_state, wind_direction_deg):
    """Score fuel cells downwind of active fire.

    FireSim treats wind_direction_deg as the direction the wind travels
    toward, so the same convention is used here.  The score is intentionally
    local and relative; it is not a geographic lookup.
    """
    active = np.argwhere(_active_mask(fire_state))
    if not len(active) or not len(rows):
        return np.zeros(len(rows), dtype=np.float32)
    theta = math.radians(float(wind_direction_deg))
    wind = np.array([math.sin(theta), math.cos(theta)], dtype=np.float32)
    points = np.column_stack((rows, cols)).astype(np.float32)
    best = np.zeros(len(points), dtype=np.float32)
    for fire_row, fire_col in active[:: max(1, len(active) // 256)]:
        delta = points - np.array([fire_row, fire_col], dtype=np.float32)
        distance = np.linalg.norm(delta, axis=1) + 1e-6
        alignment = (delta @ wind) / distance
        best = np.maximum(best, np.maximum(alignment, 0.0) / distance)
    return best


def _zone_stats(env, fire_state):
    active = _active_mask(fire_state)
    adjacent_fuel = _eight_connected_dilation(active) & (fire_state == FUEL)
    building = env.grid_static[LAYER_INDEX["building_density"]] > BUILDING_PRESENCE_THRESHOLD
    population = env.grid_static[LAYER_INDEX["population_density"]]
    threatened = building & active
    stats = []
    for zone in env.zones:
        r0, r1 = zone["row_range"]
        c0, c1 = zone["col_range"]
        state = fire_state[r0:r1, c0:c1]
        active_zone = active[r0:r1, c0:c1]
        fuel_zone = adjacent_fuel[r0:r1, c0:c1]
        threat_zone = threatened[r0:r1, c0:c1]
        pop_zone = population[r0:r1, c0:c1]
        stats.append({
            "active": int(active_zone.sum()),
            "adjacent_fuel": int(fuel_zone.sum()),
            "threatened_population": float(pop_zone[threat_zone].max()) if threat_zone.any() else 0.0,
            "fuel": float((state == FUEL).mean()),
            "building": float(building[r0:r1, c0:c1].mean()),
            "row": float(zone["centroid_row"]),
            "col": float(zone["centroid_col"]),
        })
    return stats, adjacent_fuel, threatened


def resolve_relative_targets(env, obs):
    """Return target zone ids and features for every resource/candidate.

    Returns ``zones`` with shape ``(4, 6)`` and ``features`` with shape
    ``(4, 6, TARGET_FEATURE_DIM)``.  ``-1`` denotes an invalid/no-op target.
    The resource dimension matters because nearest reachable fire depends on
    routing and resource type.
    """
    fire_state = np.asarray(obs["grid"][-1])
    wind_direction = float(obs["scalars"]["wind_direction_deg"])
    stats, adjacent_fuel, threatened = _zone_stats(env, fire_state)
    active_zones = [i for i, s in enumerate(stats) if s["active"] > 0]
    fuel_zones = [i for i, s in enumerate(stats) if s["adjacent_fuel"] > 0 and s["active"] == 0]
    threat_zones = [i for i, s in enumerate(stats) if s["threatened_population"] > 0]

    active = _active_mask(fire_state)
    fuel_rows, fuel_cols = np.where(adjacent_fuel)
    downwind = _downwind_score(fuel_rows, fuel_cols, fire_state, wind_direction)
    downwind_zone = -1
    if len(downwind):
        scores = np.zeros(env.n_zones, dtype=np.float32)
        for row, col, score in zip(fuel_rows, fuel_cols, downwind):
            zone = next(z for z in env.zones if z["row_range"][0] <= row < z["row_range"][1]
                        and z["col_range"][0] <= col < z["col_range"][1])
            scores[zone["zone_id"]] += score
        eligible = [z for z in np.argsort(scores)[::-1] if stats[int(z)]["active"] == 0 and scores[z] > 0]
        downwind_zone = int(eligible[0]) if eligible else -1

    active_zone = max(active_zones, key=lambda z: stats[z]["active"], default=-1)
    adjacent_zone = max(fuel_zones, key=lambda z: stats[z]["adjacent_fuel"], default=-1)
    population_zone = max(threat_zones, key=lambda z: stats[z]["threatened_population"], default=-1)

    zones = np.full((len(RESOURCE_TYPES), N_TARGET_TYPES), -1, dtype=np.int64)
    features = np.zeros((len(RESOURCE_TYPES), N_TARGET_TYPES, TARGET_FEATURE_DIM), dtype=np.float32)
    for resource_idx, resource_type in enumerate(RESOURCE_TYPES):
        nearest = min(
            active_zones,
            key=lambda z: env.zone_travel_time_s[resource_type][z],
            default=-1,
        )
        selected = (active_zone, downwind_zone, adjacent_zone, population_zone, nearest, -1)
        for target_idx, zone_id in enumerate(selected):
            zones[resource_idx, target_idx] = zone_id
            if zone_id < 0:
                continue
            s = stats[zone_id]
            travel = env.zone_travel_time_s[resource_type][zone_id]
            if not math.isfinite(travel):
                zones[resource_idx, target_idx] = -1
                continue
            features[resource_idx, target_idx] = np.array([
                1.0,
                min(s["active"] / 256.0, 1.0),
                min(s["adjacent_fuel"] / 256.0, 1.0),
                s["threatened_population"],
                s["fuel"],
                s["building"],
                min(travel / 3600.0, 1.0),
                s["row"] / max(1.0, fire_state.shape[0]),
                s["col"] / max(1.0, fire_state.shape[1]),
                1.0 if s["active"] > 0 else 0.0,
            ], dtype=np.float32)
    return zones, features


def decode_action(resource_idx, target_idx, target_zones):
    """Convert a sampled v8 action into the unchanged environment action."""
    if target_idx == TARGET_TYPES.index("noop"):
        return None
    zone_id = int(target_zones[resource_idx, target_idx])
    if zone_id < 0:
        return None
    return RESOURCE_TYPES[resource_idx], zone_id
