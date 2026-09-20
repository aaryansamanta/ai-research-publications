"""
Gym-style RL environment wrapper around fire_sim.py.

This module defines the INTERFACE the future CNN/MLP/actor-critic model code
will train against (observation format, action format, reward). It contains
no neural networks and no training loop -- reset()/step() just drive the
deterministic fire-spread cellular automaton (fire_sim.FireSim) plus simple,
explicit resource-dispatch/reward bookkeeping.

Action space
------------
action = (resource_type, target_zone) where:
  - resource_type: one of RESOURCE_TYPES ("water_team", "trench_crew",
    "rescue_vehicle", "helicopter"), or None/"noop" to dispatch nothing this
    tick.
  - target_zone: int zone id in [0, n_zones); ignored if resource_type is
    None/"noop". Zones are a coarse rectangular partition of the grid (see
    _build_zones) used ONLY to give the action space a manageable size and a
    real travel time -- it does not change what the CNN branch sees, which
    is still the full-resolution grid.

Resources dispatch from a real, multi-station LAFD roster (see
data_pipeline/real_depots.json) -- each real station carries its own roster
across possibly multiple resource types (e.g. Station 19/Brentwood carries
both a real Engine and a real Brush Patrol unit), and a resource type can
exist at more than one station (e.g. water_team units at both Station 69
and Station 37), matching how LAFD actually pools units across stations
during a real incident rather than one station being "the" source for a
resource category city-wide. _try_dispatch() picks the nearest AVAILABLE
unit of the requested type across every station that carries it -- real
"nearest unit of type X," not a fixed single depot. Station selection is
automatic/internal to dispatch, not part of the action space: the agent
still only ever picks (resource_type, target_zone), exactly as before.

The three ground types (water_team, trench_crew, rescue_vehicle) travel via
the real road graph, one Dijkstra tree per STATION (not per type -- a
station's tree is reused for every resource type in its roster). helicopter
is the odd one out: its one station (Air Operations, Van Nuys) has no road
route at all -- travel time is straight-line distance to the target divided
by a fixed cruise speed (HELICOPTER_SPEED_MPS). That is its real advantage
(it can reach zones the road graph can't, e.g. the 1 zone flagged
unreachable for ground units), traded off against a long reload period
after each drop (HELICOPTER_RELOAD_TICKS, a round trip back to Van Nuys to
refill retardant/water -- longer than the ground units' post-arrival
DEPLOYED_BUSY_TICKS) -- a helicopter is not simply a strictly-better ground
unit, matching how limited/weather-constrained real LAFD air support is.

Observation space
-----------------
obs = {
    "grid": float32 array (n_static_layers + 1, height, width) -- the 8
        static layers from grid_static.npy (see STATIC_LAYER_NAMES) stacked
        with the current per-cell fire state (fire_sim.FireSim.state, raw
        ordinal 0-4: Safe/Fuel/Threat/Blaze/Burned Out) as the last channel.
        This is the CNN branch's input.
    "scalars": OrderedDict, fixed key order given by SCALAR_KEYS -- wind
        speed/direction, humidity, resources available per type, elapsed
        ticks. This is the MLP branch's input. Use flatten_scalars(obs) to
        get a plain float32 vector in SCALAR_KEYS order once model code
        needs one.
}

Reward (see the project reward-formula spec)
---------------------------------------------
  +50  a dispatched water_team OR helicopter's arrival extinguishes >=1
       active fire cell -- a retardant/water drop is the same suppression
       action as a ground water_team, just delivered from the air, so it
       earns the same explicit positive reward term
 -100 to -400  a building cell is consumed by fire this tick
       (BUILDING_DESTROYED_PENALTY * a population-density-based multiplier,
       1x-4x -- see POPULATION_PENALTY_SCALE/_CAP -- a building lost in
       dense Westwood costs more than an identical loss on an empty
       hillside)
  10%-50% of that (population-scaled) penalty, if the building was in a
       zone a rescue_vehicle had reached first -- the waived FRACTION itself
       scales with population_density too (RESCUE_PENALTY_REDUCTION_MIN/MAX),
       so evacuating a dense area is worth more than evacuating a sparse one
  -10  a resource action has no effect: no unit of that type was available,
       OR the unit arrives but its effect can't apply (trench_crew targets a
       cell that's already on fire; water_team/rescue_vehicle/helicopter
       target a zone with nothing to put out / no one threatened)
       (RESOURCE_WASTED_PENALTY)
  -lambda * travel_time_seconds, charged once per successful dispatch
       (LAMBDA_TRAVEL_TIME)
Congestion term intentionally omitted (Tier 3, per project plan).
Only water_team and helicopter have an explicit positive reward term --
trench_crew and rescue_vehicle create value indirectly, by preventing future
-100/-10 events, matching the reward terms specified in the project plan (no
extra terms invented here).
"""

import csv
import json
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime, timezone

import networkx as nx
import numpy as np
try:
    import osmnx as ox
except ImportError:
    ox = None
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import DATA_DIR, REAL_DEPOTS_PATH, ROADS_GRAPHML_PATH, WEATHER_CSV_PATH  # noqa: E402
from env.fire_sim import (  # noqa: E402
    BLAZE,
    BURNED_OUT,
    FUEL,
    SAFE,
    THREAT,
    FireSim,
)

GRID_STATIC_PATH = os.path.join(DATA_DIR, "grid_static.npy")
GRID_META_PATH = os.path.join(DATA_DIR, "grid_meta.json")

# Must match grid_builder.LAYER_NAMES -- duplicated here (rather than
# importing grid_builder, which pulls in geopandas/rasterio) the same way
# fire_sim.py duplicates the layer index convention.
STATIC_LAYER_NAMES = [
    "elevation", "slope", "building_density", "building_height",
    "road_mask", "fuel_density", "water_mask", "population_density",
]
LAYER_INDEX = {name: i for i, name in enumerate(STATIC_LAYER_NAMES)}

# --- Macro-zones (action-space abstraction only, see module docstring) -----
ZONE_SIZE_CELLS = 80  # -> 4x8 = 32 zones on the current 316x595 grid, ~2.4km/side

# --- Local physical footprint of a single resource's effect -----------------
# Zones exist only to make the action space small + give a real travel-time
# target; the actual water/trench/rescue effect is a small local disk at the
# zone's centroid, not the whole (much larger) zone rectangle.
EFFECT_RADIUS_CELLS = 3

# --- Ignition-point sampling heuristic ---------------------------------------
# "Sensible" default ignition = flammable, not water, AND within this many
# cells of at least one building cell -- avoids spawning episodes deep in the
# Topanga wilderness core where there's nothing for the agent to protect and
# nothing interesting to learn.
WUI_PROXIMITY_RADIUS_CELLS = 15

BUILDING_PRESENCE_THRESHOLD = 0.10  # building_density above which a cell "is a building"

# --- Fixed real-world ignition points -----------------------------------------
# TRAINING_IGNITION_POINT: the Palisades Fire's actual documented origin, near
# the Skull Rock trailhead in the Palisades Highlands, reported ~10:30am PST
# Jan 7 2025 (34.0725 N, 118.5425 W -- Wikipedia's Palisades Fire article).
# Converted to grid row/col via the grid's affine transform; confirmed a real
# fuel cell (not water/building) within WUI_PROXIMITY_RADIUS_CELLS of a
# building, same as _sample_ignition_point()'s general criterion. Used as the
# default/main training scenario (a fixed real point, not a random draw) so
# the RL agent's primary episode matches the actual event this project is
# grounded in.
TRAINING_IGNITION_POINT = (207, 222)

# VALIDATION_IGNITION_POINTS: held-out start locations elsewhere in the same
# grid, for testing whether a policy trained on TRAINING_IGNITION_POINT
# generalizes to unseen ignitions rather than memorizing one fire. Both are
# real, named chaparral hillsides at the wildland-urban interface (fuel,
# not water, within WUI_PROXIMITY_RADIUS_CELLS of a building -- confirmed the
# same way), well clear of the Palisades Highlands and the roadless
# mid-Topanga wilderness core:
#   - "mandeville_canyon": Brentwood, one grid cell (~30m) upslope from the
#     real Mandeville Canyon trailhead (34.1212 N, 118.5067 W) -- nudged off
#     the trailhead/parking footprint itself (fuel_density there is only
#     ~0.05) and into the canyon chaparral (fuel_density ~0.82).
#   - "getty_view_park": Bel-Air, next to the Getty Center (34.0987 N,
#     118.4732 W) -- a chaparral fire-road hillside that trail guides
#     describe as showing burn scarring from a past brush fire, i.e. a real,
#     precedented wildfire-prone WUI slope, not an arbitrary point.
VALIDATION_IGNITION_POINTS = {
    "mandeville_canyon": (57, 371),
    "getty_view_park": (162, 451),
}

# MULTI_IGNITION_TRAINING_SCENARIO: an ADDITIONAL, selectable curriculum
# stage (reset(scenario='multi')) -- it does not replace the single-fire
# TRAINING_IGNITION_POINT stage, which stays the default (reset(scenario=
# 'single'), or just reset()). Three real, named chaparral/WUI hillsides
# spread across the same Santa-Ana wind corridor this project is grounded
# in (Topanga -> Brentwood -> Bel-Air), modeling a severe offshore-wind
# event igniting multiple fronts near-simultaneously rather than the single
# documented Palisades Fire origin -- plausible because the real Jan 2025
# windstorm that produced the Palisades Fire also produced the Eaton Fire
# the same night, a separate simultaneous ignition elsewhere in the same
# event; a severe Santa Ana event commonly drives multiple concurrent
# ignitions across a region's exposed WUI slopes, not just one. All three
# are, like TRAINING_IGNITION_POINT/VALIDATION_IGNITION_POINTS, confirmed
# real fuel cells (not water/building) within WUI_PROXIMITY_RADIUS_CELLS of
# a building (see scratch verification: fuel_density > 0, water_mask False,
# a building cell within 15 cells):
#   - "topanga_ridge": Topanga State Park, near the real Trippet Ranch
#     trailhead (34.0958 N, 118.5844 W) -- the westernmost, highest-
#     elevation point of the corridor.
#   - "sullivan_canyon": Sullivan Canyon, Brentwood (34.0965 N, 118.5075 W)
#     -- a distinct canyon system from the mandeville_canyon validation
#     point, ~7km east of topanga_ridge.
#   - "stone_canyon": Stone Canyon Reservoir hillside, Bel-Air (34.1013 N,
#     118.4632 W) -- a distinct drainage from the getty_view_park
#     validation point (~1km away but a different canyon), ~4km east of
#     sullivan_canyon.
# The three land in three different macro-zones (see _build_zones) at
# reset, so each fire front gets its own independent nearest-fire targeting
# via _effect_target_point -- confirmed in test_inferno_env.py.
MULTI_IGNITION_TRAINING_SCENARIO = [
    (92, 118),   # topanga_ridge
    (145, 347),  # sullivan_canyon
    (159, 483),  # stone_canyon
]

# --- Resources ---------------------------------------------------------------
def _load_real_stations(path=None):
    """Load the real multi-station LAFD depot roster (see
    data_pipeline/real_depots.json for sourcing notes -- each station's
    roster of resource types is grounded in the LAFD Fire Station
    Directory, not assumed) into a list of station dicts, each with a
    'roster' {resource_type: count}. Called once at MODULE level below
    (not just lazily inside InfernoEnv._prepare_routing(), unlike the old
    single-depot version) because RESOURCE_COUNTS/RESOURCE_TYPES are
    themselves derived from it and need to exist before other modules'
    own `from env.inferno_env import RESOURCE_TYPES` resolves at THEIR
    import time (e.g. train/eval.py, train/heuristic_policy.py)."""
    with open(path or REAL_DEPOTS_PATH) as f:
        return json.load(f)["stations"]


_REAL_STATIONS = _load_real_stations()  # module-level load, ONLY for computing the constants below

RESOURCE_TYPE_ORDER = ("water_team", "trench_crew", "rescue_vehicle", "helicopter")


def _compute_resource_counts(stations):
    """Total units per resource type across all stations' rosters -- the
    real, multi-station replacement for the old one-depot-per-type flat
    RESOURCE_COUNTS constant (was a hardcoded {"water_team": 3, ...} literal;
    now derived from real_depots.json so the two can't drift out of sync).
    Order fixed by RESOURCE_TYPE_ORDER (not real_depots.json's station-list
    order) so RESOURCE_TYPES below doesn't silently depend on JSON
    ordering."""
    counts = {rtype: 0 for rtype in RESOURCE_TYPE_ORDER}
    for station in stations:
        for rtype, n in station["roster"].items():
            counts[rtype] += n
    missing = {rtype for rtype, n in counts.items() if n == 0}
    assert not missing, f"real_depots.json has no station carrying {missing}"
    return counts


RESOURCE_COUNTS = _compute_resource_counts(_REAL_STATIONS)
RESOURCE_TYPES = tuple(RESOURCE_COUNTS)
GROUND_RESOURCE_TYPES = ("water_team", "trench_crew", "rescue_vehicle")  # dispatch via road-graph routing
AIR_RESOURCE_TYPES = ("helicopter",)  # dispatch via straight-line distance, no road graph

DEFAULT_TRAFFIC_MODE = "synthetic"
ROAD_TRAFFIC_MULTIPLIER = {
    "motorway": 1.00, "trunk": 1.05, "primary": 1.15, "secondary": 1.25,
    "tertiary": 1.35, "residential": 1.20, "unclassified": 1.30, "service": 1.25,
}
ROAD_TRAFFIC_CAPACITY = {
    "motorway": 0.03, "trunk": 0.02, "primary": 0.01, "secondary": 0.01,
    "tertiary": 0.01, "residential": 0.01, "unclassified": 0.01, "service": 0.01,
}
TRAFFIC_BPR_ALPHA = 0.15
TRAFFIC_BPR_BETA = 4.0
TRAFFIC_UPDATE_INTERVAL_TICKS = 5
RESOURCE_TRAFFIC_WEIGHT = {
    "water_team": 1.0, "trench_crew": 1.0, "rescue_vehicle": 1.2, "helicopter": 0.0,
}
# Delay (in ticks) for dispatch, prep, and cooldown
RESOURCE_DELAY_CONFIG = {
    "water_team": {"dispatch_delay_ticks": 0, "arrival_setup_delay_ticks": 1, "post_effect_busy_ticks": 5},
    "trench_crew": {"dispatch_delay_ticks": 0, "arrival_setup_delay_ticks": 2, "post_effect_busy_ticks": 5},
    "rescue_vehicle": {"dispatch_delay_ticks": 0, "arrival_setup_delay_ticks": 2, "post_effect_busy_ticks": 5},
    "helicopter": {"dispatch_delay_ticks": 0, "arrival_setup_delay_ticks": 1, "post_effect_busy_ticks": 12},
}
DEPLOYED_BUSY_TICKS = RESOURCE_DELAY_CONFIG["water_team"]["post_effect_busy_ticks"]
HELICOPTER_RELOAD_TICKS = RESOURCE_DELAY_CONFIG["helicopter"]["post_effect_busy_ticks"]
HELICOPTER_SPEED_MPS = 60.0  # Real cruise speed (~135 mph) for swift aerial firefighting response

# --- Time / weather -----------------------------------------------------------
# Maps sim ticks <-> wall-clock minutes so road travel times (real seconds)
# and fire_sim ticks share a consistent clock.
TICK_DURATION_MINUTES = 2.0
MAX_TICKS = 150  # episode cap (~5 sim-hours at TICK_DURATION_MINUTES)

# Real-world anchor for tick 0: the Palisades Fire's documented ignition time,
# ~10:30am PST Jan 7 2025 near the Skull Rock trailhead in the Palisades
# Highlands (18:30 UTC = PST + 8h, no DST in January). An episode's elapsed
# sim time (tick * TICK_DURATION_MINUTES minutes) is added to this to look up
# real weather -- see _real_weather_at(). MAX_TICKS * TICK_DURATION_MINUTES =
# 300 min = 5h, so a full episode never runs past 15:30 UTC / 3:30pm PST
# Jan 7, comfortably inside the pulled Jan 7 00:00 - Jan 9 00:00 UTC window
# (see fetch_weather.py).
FIRE_START_UTC = datetime(2025, 1, 7, 18, 30, tzinfo=timezone.utc)

# --- Reward constants (see module docstring) ---------------------------------
FIRE_EXTINGUISHED_REWARD = 50.0
BUILDING_DESTROYED_PENALTY = 100.0
RESOURCE_WASTED_PENALTY = 10.0
LAMBDA_TRAVEL_TIME = 0.02  # reward penalty per second of dispatch travel time
LAMBDA_RESPONSE_DELAY = 0.02
MU_CONGESTION = 0.01

# --- New v11 reward: per-tick fire penalty and resource dispatch costs -------
# Encourages the policy to shrink the active fire front every tick, not just
# pocket the building-loss penalty and ignore the fire.  Per-dispatch costs
# prevent spam (especially helicopter spam given the 12-tick reload).
FIRE_PENALTY_PER_CELL_PER_TICK = 0.5  # positive number; subtracted as a penalty
DISPATCH_COST = {
    "water_team": 5.0,
    "trench_crew": 8.0,
    "rescue_vehicle": 3.0,
    "helicopter": 15.0,
}
# Trench crew gets a bonus per cell that reaches BURNED_OUT *without* ever
# becoming Threat or Blaze -- i.e. a firebreak held.  Pays out only on the
# tick the cell transitions Fuel/Threat-free -> BURNED_OUT.
TRENCH_BREAK_HOLD_BONUS = 2.0

# --- Population-aware building-loss penalty (population_density layer is
# 0-1, log1p+min-max normalized -- see grid_builder.py) -----------------------
# A building lost in a dense area (e.g. Westwood Village) should cost more
# than an identical loss on an empty hillside. multiplier = 1 +
# population_density * POPULATION_PENALTY_SCALE, hard-capped so a single
# building's loss can never become absurdly disproportionate even at the
# single densest cell in the whole grid.
POPULATION_PENALTY_SCALE = 3.0  # extra weight at population_density == 1.0
POPULATION_PENALTY_MULTIPLIER_CAP = 4.0  # ceiling: 1x (empty) to 4x (max density)

# RESCUE_PENALTY_REDUCTION is population-dependent rather than one flat
# fraction: dispatching a rescue_vehicle to a dense area should waive MORE of
# the (now larger) penalty than dispatching it to a sparse one, since
# rescue's whole purpose is protecting people, not property -- this is the
# most direct place population data should matter (see module docstring).
RESCUE_PENALTY_REDUCTION_MIN = 0.5  # waived fraction at population_density == 0
RESCUE_PENALTY_REDUCTION_MAX = 0.9  # waived fraction at population_density == 1


def _population_penalty_multiplier(population_density):
    return min(1.0 + population_density * POPULATION_PENALTY_SCALE, POPULATION_PENALTY_MULTIPLIER_CAP)


def _rescue_penalty_reduction(population_density):
    return RESCUE_PENALTY_REDUCTION_MIN + population_density * (
        RESCUE_PENALTY_REDUCTION_MAX - RESCUE_PENALTY_REDUCTION_MIN
    )

SCALAR_KEYS = (
    "wind_speed_mph",
    "wind_direction_deg",
    "humidity_pct",
    "water_team_available",
    "trench_crew_available",
    "rescue_vehicle_available",
    "helicopter_available",
    "time_elapsed_ticks",
    "traffic_mean_load",
    "traffic_max_load",
    "active_ground_resources",
)


def flatten_scalars(scalars):
    """Convert the observation's 'scalars' OrderedDict into a fixed-order
    float32 vector, in SCALAR_KEYS order, for the future MLP branch."""
    return np.array([scalars[k] for k in SCALAR_KEYS], dtype=np.float32)


def _placeholder_weather_schedule(tick):
    """Fixed synthetic weather used before real data was wired in, kept
    around (opt in via InfernoEnv.reset(use_real_weather=False)) purely so
    the earlier hand-validated test_fire_sim.py scenario (uphill/downhill
    spread ratio, road/water containment) stays exactly reproducible for
    debugging. Ramps a Santa-Ana-style offshore wind up over the first 10
    ticks then holds it, with humidity held low and constant -- deliberately
    simple/fixed, NOT meant to be realistic."""
    wind_speed_mph = min(10.0 + 3.5 * tick, 45.0)
    wind_direction_deg = 45.0  # from the NE, consistent with the validated test_fire_sim.py scenario
    humidity_pct = 8.0
    return wind_speed_mph, wind_direction_deg, humidity_pct


def _load_real_weather(path=WEATHER_CSV_PATH):
    """Load the real Jan 7-8 2025 ASOS time series (see fetch_weather.py)
    into parallel arrays: epoch seconds (sorted, for searchsorted) and
    (wind_speed_mph, wind_direction_deg, humidity_pct) rows."""
    epochs, values = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            epochs.append(ts.timestamp())
            values.append((float(row["wind_speed_mph"]), float(row["wind_direction_deg"]), float(row["humidity_pct"])))
    order = np.argsort(epochs)
    epochs = np.array(epochs, dtype=np.float64)[order]
    values = np.array(values, dtype=np.float64)[order]
    return epochs, values


def _real_weather_at(elapsed_seconds, weather_epochs, weather_values):
    """Step-function lookup: the most recent real observation at or before
    FIRE_START_UTC + elapsed_seconds (real METAR obs update hourly, so a
    dispatcher would be working off the latest known reading, not a smoothed
    interpolation). Holds the LAST row if elapsed_seconds runs past the end
    of the loaded series (rather than looping -- weather doesn't repeat, and
    a stale-but-real last reading is a more honest fallback than an
    arbitrary wrap-around); this is expected to never actually trigger for a
    standard MAX_TICKS episode anchored at FIRE_START_UTC (see that
    constant's comment), but protects against out-of-range access if either
    changes later."""
    target_epoch = FIRE_START_UTC.timestamp() + elapsed_seconds
    idx = np.searchsorted(weather_epochs, target_epoch, side="right") - 1
    idx = int(np.clip(idx, 0, len(weather_epochs) - 1))
    wind_speed_mph, wind_direction_deg, humidity_pct = weather_values[idx]
    return float(wind_speed_mph), float(wind_direction_deg), float(humidity_pct)


def _disk_slice(center_row, center_col, radius, height, width):
    """Bounding-box slice + local boolean disk mask (Euclidean, like
    FireSim.ignite) around (center_row, center_col), clipped to the grid."""
    r0, r1 = max(0, center_row - radius), min(height, center_row + radius + 1)
    c0, c1 = max(0, center_col - radius), min(width, center_col + radius + 1)
    rr, cc = np.ogrid[r0:r1, c0:c1]
    mask = (rr - center_row) ** 2 + (cc - center_col) ** 2 <= radius ** 2
    return r0, r1, c0, c1, mask


def _apply_water(sim, row, col):
    """Knock down active fire in a small disk around (row, col): Threat/Blaze
    cells are extinguished (-> SAFE, not BURNED_OUT, so they are never later
    miscounted as a "building destroyed" event). Returns cells extinguished."""
    r0, r1, c0, c1, disk = _disk_slice(row, col, EFFECT_RADIUS_CELLS, sim.height, sim.width)
    region = sim.state[r0:r1, c0:c1]
    active = disk & np.isin(region, (THREAT, BLAZE))
    n = int(active.sum())
    if n:
        region[active] = SAFE
    return n


def _apply_trench(sim, row, col):
    """Dig a permanent fire break in a small disk around (row, col). Fails
    (0 cells affected) if any cell in the footprint is already Threat/Blaze
    -- per the project plan, a trench crew can't place a line on ground
    that's already burning. Otherwise converts FUEL cells in the footprint
    to SAFE and permanently zeroes their ignitability."""
    r0, r1, c0, c1, disk = _disk_slice(row, col, EFFECT_RADIUS_CELLS, sim.height, sim.width)
    region_state = sim.state[r0:r1, c0:c1]
    if (disk & np.isin(region_state, (THREAT, BLAZE))).any():
        return 0
    fuel_cells = disk & (region_state == FUEL)
    n = int(fuel_cells.sum())
    if n:
        region_state[fuel_cells] = SAFE
        sim.ignitability[r0:r1, c0:c1][fuel_cells] = 0.0
    return n


def _apply_rescue(sim, row, col, evacuated_cells):
    """Mark threatened building cells in a small disk around (row, col) as
    evacuated: they still burn per fire_sim's normal physics, but the
    building-destroyed penalty is reduced later (see
    _rescue_penalty_reduction(), population-dependent) instead of the reward
    branch changing fire_sim's state directly."""
    r0, r1, c0, c1, disk = _disk_slice(row, col, EFFECT_RADIUS_CELLS, sim.height, sim.width)
    building_region = sim.building_density[r0:r1, c0:c1]
    state_region = sim.state[r0:r1, c0:c1]
    threatened = disk & (building_region > BUILDING_PRESENCE_THRESHOLD) & np.isin(state_region, (THREAT, BLAZE))
    rows, cols = np.where(threatened)
    for lr, lc in zip(rows, cols):
        evacuated_cells.add((r0 + lr, c0 + lc))
    return len(rows)


def _effect_target_point(sim, zone):
    """Where a resource's small EFFECT_RADIUS_CELLS footprint should land
    within its (much larger) target zone.

    Zones are ~80 cells / ~2.4km per side while the physical effect disk is
    only ~7 cells / ~210m across, so pinning every arrival to the zone's
    fixed geometric centroid means water/rescue would need the fire to be
    sitting on that one exact point -- effectively never, even for a policy
    that has correctly identified the right zone. Instead, target the
    centroid of whatever active fire (Threat/Blaze) cells are currently in
    the zone, if any -- i.e. "help wherever the fire actually is within the
    zone I was sent to". Falls back to the zone's geometric centroid when
    there's no fire in the zone yet, which is exactly the case a trench crew
    digging a defensive line ahead of the fire needs."""
    r0, r1 = zone["row_range"]
    c0, c1 = zone["col_range"]
    region_state = sim.state[r0:r1, c0:c1]
    active_rows, active_cols = np.where(np.isin(region_state, (THREAT, BLAZE)))
    if len(active_rows):
        return r0 + int(round(active_rows.mean())), c0 + int(round(active_cols.mean()))
    return zone["centroid_row"], zone["centroid_col"]


def _near_building_mask(building_present, radius):
    """Vectorized "is there a building-present cell within `radius` cells"
    test for every cell, via a summed-area table (avoids adding a scipy/
    scikit-image dependency just for a box-window OR)."""
    h, w = building_present.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(building_present.astype(np.int64), axis=0), axis=1)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    r0 = np.broadcast_to(np.clip(rows - radius, 0, h), (h, w))
    r1 = np.broadcast_to(np.clip(rows + radius + 1, 0, h), (h, w))
    c0 = np.broadcast_to(np.clip(cols - radius, 0, w), (h, w))
    c1 = np.broadcast_to(np.clip(cols + radius + 1, 0, w), (h, w))
    total = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    return total > 0


def _new_resource_unit(station_id):
    """station_id: which real station this unit's roster slot belongs to
    -- fixed for the unit's lifetime (a unit always returns to and
    redispatches from its own home station), used by _try_dispatch() to
    look up this specific unit's travel time to a candidate zone."""
    return {
        "state": "available", "remaining_ticks": 0, "target_zone": None,
        "station_id": station_id, "route_edges": [], "travel_time_s": 0.0,
        "traffic_delay_s": 0.0, "dispatch_tick": None, "arrival_tick": None,
        "effect_tick": None, "available_again_tick": None,
    }


def _build_zones(grid_static, meta, zone_size_cells=ZONE_SIZE_CELLS):
    height, width = meta["height"], meta["width"]
    water = grid_static[LAYER_INDEX["water_mask"]] > 0.5
    building_density = grid_static[LAYER_INDEX["building_density"]]

    zones = []
    zone_id = 0
    num_rows = 4
    num_cols = 8
    
    for r in range(num_rows):
        r0 = int(r * height / num_rows)
        r1 = int((r + 1) * height / num_rows)
        for c in range(num_cols):
            c0 = int(c * width / num_cols)
            c1 = int((c + 1) * width / num_cols)
            zones.append({
                "zone_id": zone_id,
                "row_range": (r0, r1),
                "col_range": (c0, c1),
                "centroid_row": (r0 + r1) // 2,
                "centroid_col": (c0 + c1) // 2,
                "is_water": bool(water[r0:r1, c0:c1].mean() > 0.9),
                "building_cells": int((building_density[r0:r1, c0:c1] > BUILDING_PRESENCE_THRESHOLD).sum()),
            })
            zone_id += 1
    return zones


def _load_routing_graph(grid_crs):
    """Load the OSMnx road graph and project it into the grid's CRS so zone
    centroids (already in that CRS, from the grid's affine transform) can be
    used directly for nearest-node lookups without a second pyproj hop.
    Also adds OSMnx's imputed edge speeds/travel times, reused for real
    shortest-path ETAs (nx.single_source_dijkstra_path_length, weight=
    'travel_time', seconds) from a fixed depot to each zone."""
    graph = ox.load_graphml(ROADS_GRAPHML_PATH)
    graph = ox.project_graph(graph, to_crs=grid_crs)
    graph = ox.routing.add_edge_speeds(graph)
    graph = ox.routing.add_edge_travel_times(graph)
    return graph


class InfernoEnv:
    """Gym-style environment: reset(ignition_point=None, ignition_points=None,
    scenario='single') -> obs; step(action) -> (obs, reward, done, info). See
    the module docstring for the observation/action/reward formats, and
    reset()'s own docstring for the ignition-point selection rules."""

    def __init__(self, grid_static_path=GRID_STATIC_PATH, grid_meta_path=GRID_META_PATH, seed=None,
                 traffic_mode=DEFAULT_TRAFFIC_MODE, delay_config=None):
        self.grid_static = np.load(grid_static_path).astype(np.float32)
        with open(grid_meta_path) as f:
            self.meta = json.load(f)
        self.height, self.width = self.meta["height"], self.meta["width"]
        self.rng = np.random.default_rng(seed)
        if traffic_mode not in ("legacy", "synthetic"):
            raise ValueError("traffic_mode must be 'legacy' or 'synthetic'")
        self.traffic_mode = traffic_mode
        self.delay_config = {
            rtype: dict((delay_config or RESOURCE_DELAY_CONFIG)[rtype]) for rtype in RESOURCE_TYPES
        }

        self._building_mask = self.grid_static[LAYER_INDEX["building_density"]] > BUILDING_PRESENCE_THRESHOLD

        # Reuse FireSim's own ignitability/water logic as the single source
        # of truth for "is this cell a valid fuel cell" rather than
        # re-deriving the same physics constants here.
        probe_sim = FireSim(self.grid_static, self.meta, seed=0)
        fuel_cell_mask = probe_sim.state == FUEL
        near_building = _near_building_mask(self._building_mask, WUI_PROXIMITY_RADIUS_CELLS)
        self._ignition_candidates = np.argwhere(fuel_cell_mask & near_building)
        assert len(self._ignition_candidates) > 0, "no WUI-adjacent fuel cells found for ignition sampling"

        self.zones = _build_zones(self.grid_static, self.meta)
        self.n_zones = len(self.zones)

        print(f"[InfernoEnv] Loading + preparing road routing graph from {ROADS_GRAPHML_PATH} ...")
        self.road_graph = _load_routing_graph(self.meta["crs"])
        self._prepare_routing()
        self._edge_resource_load = {}

        self._weather_epochs, self._weather_values = _load_real_weather()
        self.use_real_weather = True  # overridable per-episode via reset(use_real_weather=...)

        self.sim = None
        self.tick_count = 0
        self.resources = None
        self.evacuated_cells = set()
        self._last_weather = None
        self.ignition_point = None
        self.ignition_points = None

    def _prepare_routing(self):
        """Compute, for every STATION (v2 of this method -- was per
        resource type), a per-zone travel time in seconds from that
        station's real location to each zone's centroid. Road stations
        (travel_mode='road') get a real road-graph Dijkstra ETA; the one
        air station (travel_mode='air_straight_line', Station 114/Air
        Operations) gets straight-line distance / HELICOPTER_SPEED_MPS
        instead -- no road graph involved, so it is never math.inf the way
        an unreachable-by-road zone is for ground units.

        self._station_travel_time_s[station_id] holds the real per-station
        data _try_dispatch() uses to pick the nearest available UNIT.
        self.zone_travel_time_s[resource_type] is kept as a backward-
        compatible per-TYPE view (the best/minimum travel time to each zone
        across every station carrying that type) for callers that only
        care "how fast can type X reach zone Z" -- heuristic_policy.py and
        test_inferno_env.py's diagnostics -- without needing to know about
        individual stations."""
        transform_coeffs = self.meta["transform"]  # (a, b, c, d, e, f), (col,row)->(x,y)
        a, b, c, d, e, f = transform_coeffs

        def pixel_to_xy(row, col):
            x = a * (col + 0.5) + b * (row + 0.5) + c
            y = d * (col + 0.5) + e * (row + 0.5) + f
            return x, y

        zone_xy = [pixel_to_xy(z["centroid_row"], z["centroid_col"]) for z in self.zones]
        xs = [xy[0] for xy in zone_xy]
        ys = [xy[1] for xy in zone_xy]
        zone_nodes = ox.distance.nearest_nodes(self.road_graph, X=xs, Y=ys)
        for zone, node in zip(self.zones, zone_nodes):
            zone["road_node"] = int(node)

        transformer = Transformer.from_crs("EPSG:4326", self.meta["crs"], always_xy=True)
        # Fresh load per instance (not the module-level _REAL_STATIONS used only to
        # compute RESOURCE_COUNTS) so each InfernoEnv gets its own station dicts to
        # mutate (road_node below) -- avoids sharing mutable state across instances.
        self.stations = _load_real_stations()
        self._stations_by_type = {}
        for station in self.stations:
            for rtype in station["roster"]:
                self._stations_by_type.setdefault(rtype, []).append(station["station_id"])

        self._station_travel_time_s = {}

        for station in self.stations:
            station_id = station["station_id"]
            station_x, station_y = transformer.transform(station["lon"], station["lat"])

            if station["travel_mode"] == "air_straight_line":
                self._station_travel_time_s[station_id] = [
                    math.hypot(x - station_x, y - station_y) / HELICOPTER_SPEED_MPS for x, y in zone_xy
                ]
                continue

            station_node = ox.distance.nearest_nodes(self.road_graph, X=station_x, Y=station_y)
            station["road_node"] = int(station_node)

            travel_times_from_station = nx.single_source_dijkstra_path_length(
                self.road_graph, station_node, weight="travel_time"
            )
            self._station_travel_time_s[station_id] = [
                travel_times_from_station.get(int(node), math.inf) for node in zone_nodes
            ]

            n_unreachable = sum(1 for t in self._station_travel_time_s[station_id] if not math.isfinite(t))
            if n_unreachable:
                print(f"[InfernoEnv] Warning: {n_unreachable}/{self.n_zones} zones unreachable "
                      f"from {station['station_name']} via the road graph "
                      f"(dispatches routed to this station specifically will be wasted there).")

        self.zone_travel_time_s = {
            rtype: [
                min((self._station_travel_time_s[sid][z] for sid in self._stations_by_type.get(rtype, [])),
                    default=math.inf)
                for z in range(self.n_zones)
            ]
            for rtype in RESOURCE_TYPES
        }

    def _road_class(self, data):
        highway = data.get("highway", "unclassified")
        if isinstance(highway, (list, tuple)):
            highway = highway[0] if highway else "unclassified"
        return str(highway).lower()

    def _synthetic_background_factor(self):
        """Deterministic traffic profile for the simulated fire timeline."""
        hour = (10.5 + self.tick_count * TICK_DURATION_MINUTES / 60.0) % 24.0
        if 7.0 <= hour < 9.5 or 16.0 <= hour < 19.0:
            return 1.25
        if 11.0 <= hour < 14.0:
            return 1.05
        return 1.10

    def _edge_effective_time(self, u, v, data):
        """Weight callback for dynamic synthetic-traffic Dijkstra routing."""
        if "travel_time" in data:
            base_time = float(data["travel_time"])
            edge_data = data
            edge_key = data.get("key")
        else:
            # NetworkX passes all parallel-edge records for a MultiDiGraph to
            # a callable weight function. Select the fastest current edge.
            candidates = []
            for key, attrs in data.items():
                if not isinstance(attrs, dict):
                    continue
                candidates.append((self._edge_effective_time(u, v, {**attrs, "key": key}), attrs, key))
            if not candidates:
                return math.inf
            return min(candidates, key=lambda item: item[0])[0]

        if self.traffic_mode == "legacy":
            return base_time
        road_class = self._road_class(edge_data)
        background = ROAD_TRAFFIC_MULTIPLIER.get(road_class, ROAD_TRAFFIC_MULTIPLIER["unclassified"])
        background *= self._synthetic_background_factor()
        load = 0.0
        if edge_key is not None:
            load = self._edge_resource_load.get((u, v, edge_key), 0.0)
        capacity = ROAD_TRAFFIC_CAPACITY.get(road_class, ROAD_TRAFFIC_CAPACITY["unclassified"])
        congestion = 1.0 + TRAFFIC_BPR_ALPHA * (load / max(capacity, 1.0)) ** TRAFFIC_BPR_BETA
        return base_time * background * congestion

    def _route_edges_from_nodes(self, nodes):
        edges = []
        for u, v in zip(nodes, nodes[1:]):
            records = self.road_graph.get_edge_data(u, v) or {}
            best = min(
                records.items(),
                key=lambda item: self._edge_effective_time(u, v, {**item[1], "key": item[0]}),
                default=(None, None),
            )
            if best[0] is not None:
                edges.append((u, v, best[0]))
        return edges

    def _dynamic_ground_route(self, station_id, target_zone_id):
        station = next(s for s in self.stations if s["station_id"] == station_id)
        source = int(station["road_node"])
        target = int(self.zones[target_zone_id]["road_node"])
        try:
            travel_s, nodes = nx.single_source_dijkstra(
                self.road_graph, source, target, weight=self._edge_effective_time
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return math.inf, []
        return float(travel_s), self._route_edges_from_nodes(nodes)

    def _reserve_route(self, resource_type, route_edges):
        weight = RESOURCE_TRAFFIC_WEIGHT[resource_type]
        if weight <= 0:
            return
        for edge in route_edges:
            self._edge_resource_load[edge] = self._edge_resource_load.get(edge, 0.0) + weight

    def _release_route(self, resource_type, route_edges):
        weight = RESOURCE_TRAFFIC_WEIGHT[resource_type]
        if weight <= 0:
            return
        for edge in route_edges:
            remaining = self._edge_resource_load.get(edge, 0.0) - weight
            if remaining > 1e-9:
                self._edge_resource_load[edge] = remaining
            else:
                self._edge_resource_load.pop(edge, None)

    # --- Episode lifecycle ---------------------------------------------------

    def _sample_ignition_point(self):
        idx = self.rng.integers(len(self._ignition_candidates))
        row, col = self._ignition_candidates[idx]
        return int(row), int(col)

    def reset(self, ignition_point=None, ignition_points=None, scenario="single",
              seed=None, use_real_weather=True):
        """use_real_weather=True (default): wind/humidity come from the real
        Jan 7-8 2025 ASOS series (see _real_weather_at()). Set False to fall
        back to the old fixed Santa-Ana-ramp placeholder, e.g. to reproduce
        the originally-validated test_fire_sim.py scenario exactly.

        Ignition points -- three ways to specify them, checked in this order:
          1. ignition_points=[(row, col), ...] -- an explicit list, ignites
             all of them. Takes precedence over `scenario`.
          2. scenario='multi' (with ignition_points left as None) -- ignites
             MULTI_IGNITION_TRAINING_SCENARIO's three fixed real points.
          3. scenario='single' (the default) -- the original single-fire
             behavior, unchanged: ignites `ignition_point` if given, else
             samples one random WUI-adjacent fuel cell via
             _sample_ignition_point(). This is the default curriculum stage;
             'multi' is an additional stage on top of it, not a replacement.
        ignition_point and ignition_points are mutually exclusive.
        """
        if ignition_point is not None and ignition_points is not None:
            raise ValueError("Pass either ignition_point or ignition_points, not both")

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.use_real_weather = use_real_weather

        sim_seed = int(self.rng.integers(0, 2 ** 31 - 1))
        self.sim = FireSim(self.grid_static, self.meta, seed=sim_seed)

        if ignition_points is not None:
            points = [(int(r), int(c)) for r, c in ignition_points]
        elif scenario == "multi":
            points = list(MULTI_IGNITION_TRAINING_SCENARIO)
        elif scenario == "single":
            points = [ignition_point if ignition_point is not None else self._sample_ignition_point()]
        else:
            raise ValueError(f"Unknown scenario {scenario!r}; expected 'single' or 'multi'")

        for row, col in points:
            self.sim.ignite(row, col, radius=1)
        self.ignition_point = points[0]  # backward-compat: first/primary ignition point
        self.ignition_points = points

        self.tick_count = 0
        self.resources = {rtype: [] for rtype in RESOURCE_TYPES}
        self._edge_resource_load.clear()
        for station in self.stations:
            for rtype, count in station["roster"].items():
                self.resources[rtype].extend(_new_resource_unit(station["station_id"]) for _ in range(count))
        self.evacuated_cells = set()
        self._last_weather = self._weather_schedule(0)

        return self._build_observation()

    def _update_weather(self, tick):
        if self.use_real_weather:
            wind_speed_mph, wind_direction_deg, humidity_pct = _real_weather_at(
                self.elapsed_seconds, self._weather_epochs, self._weather_values)
        else:
            wind_speed_mph = min(10.0 + 3.5 * tick, 45.0)
            wind_direction_deg = 45.0
            humidity_pct = 8.0
        
        self.wind_speed_mph = wind_speed_mph
        self.wind_direction_deg = wind_direction_deg
        self.humidity_pct = humidity_pct

    def _weather_schedule(self, tick):
        if self.use_real_weather:
            elapsed_seconds = tick * TICK_DURATION_MINUTES * 60.0
            return _real_weather_at(elapsed_seconds, self._weather_epochs, self._weather_values)
        return _placeholder_weather_schedule(tick)

    def _parse_actions(self, actions):
        if not isinstance(actions, list):
            raise ValueError("actions must be a list of (resource_type, target_zone) pairs")
        parsed = []
        for action in actions:
            if action is None:
                parsed.append((None, None))
                continue
            if not isinstance(action, tuple) or len(action) != 2:
                raise ValueError("each action must be a (resource_type, target_zone) tuple or None")
            resource_type, target_zone = action
            if resource_type is None or resource_type == "noop":
                parsed.append((None, None))
                continue
            if resource_type not in RESOURCE_TYPES:
                raise ValueError(f"Unknown resource_type {resource_type!r}; expected one of {RESOURCE_TYPES}")
            if not isinstance(target_zone, (int, np.integer)) or not (0 <= target_zone < self.n_zones):
                raise ValueError(f"target_zone {target_zone} out of range [0, {self.n_zones})")
            parsed.append((resource_type, int(target_zone)))
        return parsed

    def _try_dispatch(self, resource_type, target_zone_id):
        """v2: picks the nearest AVAILABLE unit of resource_type across
        EVERY station that carries it (self._stations_by_type), not the
        first idle unit in a flat single-depot list. "zone_unreachable"
        now means unreachable from every station carrying this type (a
        real improvement over the old single-depot version -- a zone the
        nearest station's road can't reach may still be reachable from a
        second, farther station of the same type); "no_unit_available"
        means at least one station could reach the zone but every unit
        stationed there (or at any other reachable station of this type)
        is currently busy."""
        station_ids = self._stations_by_type.get(resource_type, [])
        roster = self.resources[resource_type]
        best_unit, best_travel_s, best_route = None, math.inf, []
        best_free_flow_s = math.inf
        for unit in roster:
            if unit["state"] != "available":
                continue
            if resource_type in GROUND_RESOURCE_TYPES and self.traffic_mode == "synthetic":
                travel_s, route_edges = self._dynamic_ground_route(unit["station_id"], target_zone_id)
                free_flow_s = self._station_travel_time_s[unit["station_id"]][target_zone_id]
            else:
                travel_s = self._station_travel_time_s[unit["station_id"]][target_zone_id]
                route_edges = []
                free_flow_s = travel_s
            if math.isfinite(travel_s) and travel_s < best_travel_s:
                best_unit, best_travel_s, best_route, best_free_flow_s = unit, travel_s, route_edges, free_flow_s

        if best_unit is None:
            zone_reachable = any(
                math.isfinite(self._station_travel_time_s[sid][target_zone_id]) for sid in station_ids
            )
            return {"resource_type": resource_type, "target_zone": target_zone_id,
                    "status": "zone_unreachable" if not zone_reachable else "no_unit_available",
                    "reward_delta": -RESOURCE_WASTED_PENALTY}

        eta_ticks = max(1, math.ceil(best_travel_s / (TICK_DURATION_MINUTES * 60.0)))
        delay = self.delay_config[resource_type]
        dispatch_delay_ticks = delay["dispatch_delay_ticks"] if self.traffic_mode == "synthetic" else 0
        best_unit["state"] = "preparing" if dispatch_delay_ticks else "traveling"
        best_unit["remaining_ticks"] = dispatch_delay_ticks or eta_ticks
        best_unit["target_zone"] = target_zone_id
        best_unit["route_edges"] = best_route
        best_unit["travel_time_s"] = best_travel_s
        best_unit["traffic_delay_s"] = max(0.0, best_travel_s - best_free_flow_s)
        best_unit["pending_travel_ticks"] = eta_ticks
        best_unit["dispatch_tick"] = self.tick_count
        return {"resource_type": resource_type, "target_zone": target_zone_id, "status": "dispatched",
                "travel_time_s": best_travel_s, "traffic_delay_s": best_unit["traffic_delay_s"],
                "response_delay_ticks": dispatch_delay_ticks,
                "eta_ticks": dispatch_delay_ticks + eta_ticks,
                "station_id": best_unit["station_id"],
                "reward_delta": -LAMBDA_TRAVEL_TIME * best_travel_s
                - LAMBDA_RESPONSE_DELAY * dispatch_delay_ticks * TICK_DURATION_MINUTES * 60.0}

    def _advance_resources(self):
        """Advance every non-available unit by one tick; apply effects for
        units that just finished traveling, and free up units that just
        finished their post-arrival busy period."""
        reward = 0.0
        events = []
        for rtype in RESOURCE_TYPES:
            for unit in self.resources[rtype]:
                if unit["state"] == "available":
                    continue
                unit["remaining_ticks"] -= 1
                if unit["remaining_ticks"] > 0:
                    continue

                if unit["state"] == "preparing":
                    unit["state"] = "traveling"
                    unit["remaining_ticks"] = unit["pending_travel_ticks"]
                    if rtype in GROUND_RESOURCE_TYPES and self.traffic_mode == "synthetic":
                        self._reserve_route(rtype, unit["route_edges"])
                    continue

                if unit["state"] == "traveling":
                    if rtype in GROUND_RESOURCE_TYPES and self.traffic_mode == "synthetic":
                        self._release_route(rtype, unit["route_edges"])
                    unit["arrival_tick"] = self.tick_count
                    setup_ticks = self.delay_config[rtype]["arrival_setup_delay_ticks"] \
                        if self.traffic_mode == "synthetic" else 0
                    if setup_ticks:
                        unit["state"] = "arrival_setup"
                        unit["remaining_ticks"] = setup_ticks
                        continue

                    zone = self.zones[unit["target_zone"]]
                    row, col = _effect_target_point(self.sim, zone)
                    if rtype == "water_team" or rtype == "helicopter":
                        # Same suppression physics for a ground water_team and a
                        # helicopter's retardant/water drop -- only how they got
                        # there (and how long they're tied up after) differs.
                        n_affected = _apply_water(self.sim, row, col)
                    elif rtype == "trench_crew":
                        n_affected = _apply_trench(self.sim, row, col)
                    else:
                        n_affected = _apply_rescue(self.sim, row, col, self.evacuated_cells)

                    success = n_affected > 0
                    if success and (rtype == "water_team" or rtype == "helicopter"):
                        reward += FIRE_EXTINGUISHED_REWARD
                    elif not success:
                        reward -= RESOURCE_WASTED_PENALTY
                    events.append({"resource_type": rtype, "zone": unit["target_zone"],
                                   "cells_affected": n_affected, "success": success,
                                   "row": int(row), "col": int(col)})
                    unit["effect_tick"] = self.tick_count
                    unit["state"] = "deployed"
                    unit["remaining_ticks"] = self.delay_config[rtype]["post_effect_busy_ticks"] \
                        if self.traffic_mode == "synthetic" else (
                            HELICOPTER_RELOAD_TICKS if rtype == "helicopter" else DEPLOYED_BUSY_TICKS
                        )
                elif unit["state"] == "arrival_setup":
                    zone = self.zones[unit["target_zone"]]
                    row, col = _effect_target_point(self.sim, zone)
                    if rtype == "water_team" or rtype == "helicopter":
                        n_affected = _apply_water(self.sim, row, col)
                    elif rtype == "trench_crew":
                        n_affected = _apply_trench(self.sim, row, col)
                    else:
                        n_affected = _apply_rescue(self.sim, row, col, self.evacuated_cells)
                    success = n_affected > 0
                    if success and (rtype == "water_team" or rtype == "helicopter"):
                        reward += FIRE_EXTINGUISHED_REWARD
                    elif not success:
                        reward -= RESOURCE_WASTED_PENALTY
                    events.append({"resource_type": rtype, "zone": unit["target_zone"],
                                   "cells_affected": n_affected, "success": success,
                                   "row": int(row), "col": int(col)})
                    unit["effect_tick"] = self.tick_count
                    unit["state"] = "deployed"
                    unit["remaining_ticks"] = self.delay_config[rtype]["post_effect_busy_ticks"]
                elif unit["state"] == "deployed":
                    unit["state"] = "available"
                    unit["target_zone"] = None
                    unit["route_edges"] = []
                    unit["available_again_tick"] = self.tick_count
        return reward, events

    def _score_building_destruction(self, before_state):
        """Population-aware building-loss penalty -- see
        POPULATION_PENALTY_SCALE/_CAP and RESCUE_PENALTY_REDUCTION_MIN/MAX in
        the module constants. Returns per-event detail (not just aggregates)
        so callers can verify the scaling is actually doing something
        (see test_inferno_env.py)."""
        newly_burned = self._building_mask & (before_state != BURNED_OUT) & (self.sim.state == BURNED_OUT)
        rows, cols = np.where(newly_burned)
        reward = 0.0
        n_evacuated = 0
        events = []
        for r, c in zip(rows, cols):
            density = float(self.grid_static[LAYER_INDEX["population_density"], r, c])
            multiplier = _population_penalty_multiplier(density)
            penalty = BUILDING_DESTROYED_PENALTY * multiplier
            evacuated = (r, c) in self.evacuated_cells
            if evacuated:
                reduction = _rescue_penalty_reduction(density)
                applied_penalty = penalty * (1.0 - reduction)
                n_evacuated += 1
                self.evacuated_cells.discard((r, c))
            else:
                applied_penalty = penalty
            reward -= applied_penalty
            events.append({
                "row": int(r), "col": int(c), "population_density": density,
                "multiplier": multiplier, "evacuated": evacuated, "penalty_applied": applied_penalty,
            })
        return reward, int(len(rows)), n_evacuated, events

    def step(self, action):
        if not isinstance(action, list):
            raise ValueError("actions must be a list of (resource_type, target_zone) pairs")
        actions_list = action

        # Advance units already en route/deployed BEFORE committing this
        # tick's new dispatches.
        reward, resource_events = self._advance_resources()

        dispatch_infos = []
        for act in actions_list:
            resource_type, target_zone = self._parse_actions([act])[0]
            if resource_type is not None:
                d_info = self._try_dispatch(resource_type, target_zone)
                reward += d_info["reward_delta"]
                dispatch_infos.append(d_info)
        dispatch_cost_total = sum(
            DISPATCH_COST[d["resource_type"]] for d in dispatch_infos if d["status"] == "dispatched"
        )

        wind_speed, wind_direction, humidity = self._weather_schedule(self.tick_count)
        self._last_weather = (wind_speed, wind_direction, humidity)

        before_state = self.sim.state.copy()
        # v11 trench-hold bonus uses pre_trench_mask: cells protected
        # (ignitability <= 0) and still Safe/Fuel at the START of this tick.
        pre_trench_mask = (self.sim.ignitability <= 0) & np.isin(before_state, (SAFE, FUEL))
        self.sim.step(wind_speed_mph=wind_speed, wind_direction_deg=wind_direction, humidity_pct=humidity)
        destroy_reward, n_destroyed, n_destroyed_evacuated, destruction_events = \
            self._score_building_destruction(before_state)
        reward += destroy_reward

        self.tick_count += 1
        counts = self.sim.state_counts()
        # v11: per-tick fire penalty for Threat + Blaze cells (Blaze burns fuel
        # now, Threat is about to burn next tick -- both should hurt).
        active_fire_cells = counts["Threat"] + counts["Blaze"]
        fire_penalty = FIRE_PENALTY_PER_CELL_PER_TICK * active_fire_cells

        # v11: trench-hold bonus.  A cell that was protected (ignitability <= 0
        # and still Safe/Fuel at the start of this tick) and ends this tick in
        # BURNED_OUT without ever having been Threat or Blaze is a firebreak
        # that held through the fire's arrival.  These cells are the rare but
        # high-signal reward that makes trench crew worth dispatching
        # proactively.
        post_state = self.sim.state
        held_now_burned = pre_trench_mask & (post_state == BURNED_OUT)
        trench_bonus = TRENCH_BREAK_HOLD_BONUS * int(held_now_burned.sum())

        reward_components = {
            "fire_penalty": -float(fire_penalty),
            "dispatch_cost": -float(dispatch_cost_total),
            "trench_bonus": float(trench_bonus),
            "buildings_destroyed": float(destroy_reward),
            "travel_delay": float(sum(d["reward_delta"] for d in dispatch_infos)),
            "wasted": float(-RESOURCE_WASTED_PENALTY * sum(1 for d in dispatch_infos if d["status"] != "dispatched")),
        }
        # Fire-extinguished component comes from _advance_resources (already in
        # `reward`).  Capture it by reading the diff of resource events.
        extinguish_count = sum(
            1 for ev in resource_events
            if ev.get("success") and ev["resource_type"] in ("water_team", "helicopter")
        )
        reward_components["fire_extinguished"] = float(FIRE_EXTINGUISHED_REWARD * extinguish_count)

        reward = reward - fire_penalty - dispatch_cost_total + trench_bonus

        contained = counts["Threat"] == 0 and counts["Blaze"] == 0
        timed_out = self.tick_count >= 1000  # High cap allowing simulation to run until fully contained or burned out
        done = contained or timed_out

        info = {
            "dispatch": dispatch_infos,
            "resource_events": resource_events,
            "weather": {"wind_speed_mph": wind_speed, "wind_direction_deg": wind_direction, "humidity_pct": humidity},
            "state_counts": counts,
            "buildings_destroyed": n_destroyed,
            "buildings_destroyed_evacuated": n_destroyed_evacuated,
            "building_destruction_events": destruction_events,
            "contained": contained,
            "timeout": timed_out and not contained,
            "tick": self.tick_count,
            "active_fire_cells": active_fire_cells,
            "trench_held": int(held_now_burned.sum()),
            "reward_components": reward_components,
        }
        return self._build_observation(), reward, done, info

    # --- Observation -----------------------------------------------------------

    def _build_observation(self):
        grid = np.concatenate(
            [self.grid_static, self.sim.state[np.newaxis, :, :].astype(np.float32)], axis=0
        )
        wind_speed, wind_direction, humidity = self._last_weather
        available = {
            rtype: float(sum(1 for u in self.resources[rtype] if u["state"] == "available"))
            for rtype in RESOURCE_TYPES
        }
        loads = list(self._edge_resource_load.values())
        active_ground = sum(
            unit["state"] == "traveling"
            for rtype in GROUND_RESOURCE_TYPES for unit in self.resources[rtype]
        )
        scalars = OrderedDict([
            ("wind_speed_mph", wind_speed),
            ("wind_direction_deg", wind_direction),
            ("humidity_pct", humidity),
            ("water_team_available", available["water_team"]),
            ("trench_crew_available", available["trench_crew"]),
            ("rescue_vehicle_available", available["rescue_vehicle"]),
            ("helicopter_available", available["helicopter"]),
            ("time_elapsed_ticks", float(self.tick_count)),
            ("traffic_mean_load", float(np.mean(loads)) if loads else 0.0),
            ("traffic_max_load", float(max(loads)) if loads else 0.0),
            ("active_ground_resources", float(active_ground)),
        ])
        return {"grid": grid, "scalars": scalars}
