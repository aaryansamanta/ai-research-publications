"""
Numerical validation of fire_sim.py's simulated fire spread against the real
Palisades Fire's actual burned extent (WFIGS/LA GeoHub perimeter data via
fetch_perimeter.py). Flagged as a known gap in the project since the grid/
fire-spread model was built -- this is that validation, run honestly: the
sim's parameters (BASE_SPREAD_PROB, SLOPE_COEFF, WIND_COEFF, etc. in
fire_sim.py) are NOT touched or tuned based on these results, before or
after seeing them. Whatever the numbers say, they're reported as-is.

What this does and does NOT claim to measure
----------------------------------------------
This runs the RAW fire_sim.py cellular automaton -- no RL agent, no
heuristic policy, NO SUPPRESSION RESOURCES AT ALL -- from
TRAINING_IGNITION_POINT (the real documented Skull Rock origin), driven by
the real Jan 7-9 2025 KSMO weather series already used elsewhere in this
project. The real Palisades Fire's actual final perimeter reflects a massive
multi-agency firefighting response (structure defense, hand crews, dozers,
aircraft) that this project's tiny 4-resource-type/32-zone RL abstraction
doesn't remotely model, and this validation run has ZERO suppression of any
kind. So this is NOT a test of "does the RL agent's dispatch produce a
realistic outcome" -- it is a test of "does the underlying spread PHYSICS
(fuel/slope/wind/humidity/road/ember-spotting rules) produce a plausible
burned footprint shape and scale, independent of suppression." A simulated
extent that OVER-shoots the real (suppressed) perimeter is the expected
direction of any discrepancy for exactly that reason, not necessarily a sign
the physics model itself is wrong -- this is stated up front rather than
discovered and spun after the fact.

Two further honest, unavoidable limitations, stated before any numbers:
  1. Real KSMO weather data was only pulled for Jan 7 00:00 - Jan 9 00:00 UTC
     (see fetch_weather.py). The only real perimeter snapshot found publicly
     available (see fetch_perimeter.py) is dated ~13.5 days later, Jan 21
     2025. To reach that comparison point at all, this script runs fire_sim
     for the full ~13.5 days using inferno_env._real_weather_at()'s EXISTING
     hold-last-value behavior once the real series runs out (~tick 450 of
     ~9,525) -- i.e. real Santa Ana conditions for the first ~2 days, then a
     frozen Jan-8/9 snapshot for the remaining ~11.5 days, not genuinely
     evolving real weather for that whole window. Any real Palisades Fire
     progression driven by weather changes after Jan 9 (the event's Santa
     Ana winds did NOT sustain at their peak for two more weeks) is
     therefore not represented.
  2. The Jan 21 perimeter is itself a "dissolved" single snapshot (21
     disjoint polygons merged from NIFC's daily perimeters, per LA County's
     own dataset description) -- not a timestamped progression. So this is a
     single-point comparison (simulated cumulative burned area after ~13.5
     sim-days vs. the one real snapshot at that same point), not a multi-
     timestamp IoU curve, despite that being the original hope -- no
     sub-14-day-granularity public perimeter data for this specific fire
     could be located (checked: WFIGS's current-year rolling layers, which
     clear after a fire's incident is closed out; the multi-decade
     WFIGS_Interagency_Perimeters history layer, whose own documentation
     states it only covers "thru the 2024 fire season").

Metrics reported: IoU (intersection-over-union of the two boolean burned-
area masks on the project's 30m grid), simulated-burned-area / real-burned-
area ratio, precision (fraction of simulated burned cells that were really
burned) and recall (fraction of really-burned cells the simulation also
burned) -- IoU alone can be a harsh, easily-misread single number for two
differently-shaped regions of similar size, so all four are reported
together for an honest full picture.

    python -m src.validation.wfigs_perimeter_validation
"""

import os
import sys
import time
from datetime import datetime, timezone

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
import rasterio.transform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import DATA_DIR, PROJECT_ROOT  # noqa: E402
from data_pipeline.fetch_perimeter import PERIMETER_PATH, fetch_perimeter  # noqa: E402
from env.fire_sim import BURNED_OUT, BLAZE, THREAT, FireSim  # noqa: E402
from env.inferno_env import (  # noqa: E402
    FIRE_START_UTC,
    GRID_META_PATH,
    GRID_STATIC_PATH,
    TICK_DURATION_MINUTES,
    TRAINING_IGNITION_POINT,
    WEATHER_CSV_PATH,
    _load_real_weather,
    _real_weather_at,
)
import json  # noqa: E402

# The perimeter snapshot's own label is a date only ("as of 20250121"), no
# time-of-day -- midnight UTC is the plainest reading of a bare date and is
# used as-is, not adjusted to make the comparison land more favorably.
TARGET_UTC = datetime(2025, 1, 21, 0, 0, tzinfo=timezone.utc)

REPORT_PATH = os.path.join(PROJECT_ROOT, "logs", "wfigs_validation_report.json")
OVERLAY_PLOT_PATH = os.path.join(DATA_DIR, "wfigs_validation_overlay.png")


def plot_overlay(sim_mask, real_mask, iou, area_ratio, recall):
    """4-color categorical map: agree-unburned / sim-only (over-prediction,
    expected given no suppression) / real-only (missed) / agree-burned
    (true positive) -- a single glance shows where the unsuppressed sim's
    footprint diverges from the real, suppressed one, not just the summary
    numbers."""
    category = np.zeros(sim_mask.shape, dtype=np.uint8)
    category[sim_mask & ~real_mask] = 1  # sim-only (over-prediction)
    category[~sim_mask & real_mask] = 2  # real-only (missed)
    category[sim_mask & real_mask] = 3  # both (true positive)
    colors = ["#eeeeee", "#f4a259", "#5b8dee", "#8b2f2f"]
    labels = ["Neither burned", "Sim only (over-prediction)", "Real only (missed)", "Both (agree)"]

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    ax.imshow(category, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(
        f"WFIGS validation: unsuppressed fire_sim.py vs. real Palisades Fire perimeter "
        f"(2025-01-21 snapshot)\nIoU={iou:.3f}  recall={recall:.3f}  area_ratio={area_ratio:.3f} "
        f"(sim / real-clipped-to-grid)"
    )
    ax.set_xlabel("grid column")
    ax.set_ylabel("grid row")
    handles = [plt.matplotlib.patches.Patch(color=c, label=lab) for c, lab in zip(colors, labels)]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OVERLAY_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved overlap visualization to: {OVERLAY_PLOT_PATH}")


def rasterize_real_perimeter(grid_meta, perimeter_path=None):
    """Reproject the real perimeter polygons to the grid's own CRS/transform
    and rasterize onto the exact same (height, width) grid fire_sim runs on,
    so the two boolean masks are directly comparable cell-for-cell.
    perimeter_path defaults to the Jan-21 snapshot (PERIMETER_PATH); pass an
    alternate path (e.g. data/palisades_perimeter_20250111.geojson) to
    validate against a different real snapshot -- see the two-timestamp
    check in project memory / the module docstring's follow-up note."""
    perimeter_path = perimeter_path or PERIMETER_PATH
    if not os.path.exists(perimeter_path):
        fetch_perimeter()
    gdf = gpd.read_file(perimeter_path)
    gdf = gdf.to_crs(grid_meta["crs"])
    transform = rasterio.transform.Affine(*grid_meta["transform"])
    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty],
        out_shape=(grid_meta["height"], grid_meta["width"]),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask.astype(bool)


def run_long_fire_sim(grid_static, grid_meta, n_ticks, weather_epochs, weather_values, seed=0):
    """Bare fire_sim.py, TRAINING_IGNITION_POINT, real weather (held-last
    past the loaded series, matching inferno_env._real_weather_at()'s own
    existing behavior) -- see module docstring. Returns the cumulative
    "ever ignited" boolean mask (THREAT/BLAZE/BURNED_OUT at any tick;
    monotonic in fire_sim.py with no suppression in play, but tracked as a
    running OR each tick rather than assumed, for certainty)."""
    sim = FireSim(grid_static, grid_meta, seed=seed)
    sim.ignite(*TRAINING_IGNITION_POINT)
    ever_burned = np.zeros((grid_meta["height"], grid_meta["width"]), dtype=bool)
    t0 = time.perf_counter()
    for tick in range(n_ticks):
        elapsed_seconds = tick * TICK_DURATION_MINUTES * 60.0
        wind_speed, wind_dir, humidity = _real_weather_at(elapsed_seconds, weather_epochs, weather_values)
        sim.step(wind_speed_mph=wind_speed, wind_direction_deg=wind_dir, humidity_pct=humidity)
        ever_burned |= (sim.state == THREAT) | (sim.state == BLAZE) | (sim.state == BURNED_OUT)
        if (tick + 1) % 500 == 0 or tick == n_ticks - 1:
            elapsed_wall = time.perf_counter() - t0
            counts = sim.state_counts()
            print(f"  tick {tick + 1}/{n_ticks}  (wall {elapsed_wall:.1f}s)  "
                  f"wind={wind_speed:.1f}mph@{wind_dir:.0f}deg  humidity={humidity:.1f}%  "
                  f"ever_burned_cells={int(ever_burned.sum())}  state_counts={counts}",
                  flush=True)
    return ever_burned, sim


def main():
    print("=== WFIGS perimeter validation ===")
    print(f"Ignition point: {TRAINING_IGNITION_POINT}  Fire start (real): {FIRE_START_UTC.isoformat()}")
    print(f"Target comparison timestamp: {TARGET_UTC.isoformat()} (perimeter snapshot's own dated label)")

    elapsed_seconds_total = (TARGET_UTC - FIRE_START_UTC).total_seconds()
    n_ticks = int(round(elapsed_seconds_total / (TICK_DURATION_MINUTES * 60.0)))
    print(f"Elapsed real time: {elapsed_seconds_total / 3600.0:.1f} hours -> {n_ticks} sim ticks "
          f"at {TICK_DURATION_MINUTES}-min/tick (vs. the RL env's MAX_TICKS=150 episode cap -- "
          f"this validation run is NOT bounded by that, see module docstring)")

    grid_static = np.load(GRID_STATIC_PATH).astype(np.float32)
    with open(GRID_META_PATH) as f:
        grid_meta = json.load(f)

    weather_epochs, weather_values = _load_real_weather(WEATHER_CSV_PATH)
    weather_end = datetime.fromtimestamp(weather_epochs[-1], tz=timezone.utc)
    ticks_with_real_weather = int((weather_epochs[-1] - FIRE_START_UTC.timestamp()) / (TICK_DURATION_MINUTES * 60.0))
    print(f"Real weather data available through: {weather_end.isoformat()} "
          f"(~tick {ticks_with_real_weather} of {n_ticks} -- weather HELD LAST beyond that, see module docstring)")

    print("\nRasterizing real perimeter onto the project grid...")
    real_mask = rasterize_real_perimeter(grid_meta)
    real_burned_cells = int(real_mask.sum())
    cell_area_acres = (grid_meta["cell_size_m"] ** 2) / 4046.8564224
    print(f"Real perimeter: {real_burned_cells} cells = {real_burned_cells * cell_area_acres:,.1f} acres "
          f"on this grid (documented real final size: ~23,400 acres -- grid clipping/rasterization "
          f"at 30m resolution accounts for any difference from the raw polygon area check in "
          f"fetch_perimeter.py's own sanity print)")

    print(f"\nRunning fire_sim for {n_ticks} ticks (this will take a while -- progress every 500 ticks)...")
    sim_mask, final_sim = run_long_fire_sim(grid_static, grid_meta, n_ticks, weather_epochs, weather_values)
    sim_burned_cells = int(sim_mask.sum())
    print(f"\nSimulated (unsuppressed) burned extent: {sim_burned_cells} cells = "
          f"{sim_burned_cells * cell_area_acres:,.1f} acres")

    intersection = int((sim_mask & real_mask).sum())
    union = int((sim_mask | real_mask).sum())
    iou = intersection / union if union > 0 else float("nan")
    precision = intersection / sim_burned_cells if sim_burned_cells > 0 else float("nan")
    recall = intersection / real_burned_cells if real_burned_cells > 0 else float("nan")
    area_ratio = sim_burned_cells / real_burned_cells if real_burned_cells > 0 else float("nan")

    print("\n=== Results (unsuppressed sim vs. real Jan-21 perimeter snapshot) ===")
    print(f"Intersection: {intersection} cells   Union: {union} cells")
    print(f"IoU: {iou:.4f}")
    print(f"Precision (of simulated-burned cells, fraction really burned): {precision:.4f}")
    print(f"Recall (of really-burned cells, fraction simulation also burned): {recall:.4f}")
    print(f"Area ratio (simulated / real): {area_ratio:.4f}")
    print("\nInterpretation reminder: this run has NO suppression modeled at all (see module docstring) "
          "-- an area_ratio > 1 (sim over-predicts extent) is the expected direction of error for that "
          "reason and is not itself evidence the spread physics are wrong; area_ratio < 1 or a low "
          "recall despite that handicap would be the more surprising/concerning result.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ignition_point": list(TRAINING_IGNITION_POINT),
        "fire_start_utc": FIRE_START_UTC.isoformat(),
        "target_utc": TARGET_UTC.isoformat(),
        "n_ticks": n_ticks,
        "ticks_with_real_weather": ticks_with_real_weather,
        "real_burned_cells": real_burned_cells,
        "real_burned_acres": real_burned_cells * cell_area_acres,
        "sim_burned_cells": sim_burned_cells,
        "sim_burned_acres": sim_burned_cells * cell_area_acres,
        "intersection_cells": intersection,
        "union_cells": union,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "area_ratio": area_ratio,
        "caveats": [
            "No suppression resources modeled at all -- pure fire_sim.py physics vs. a real, "
            "heavily-suppressed fire; area_ratio > 1 is the expected direction of error.",
            f"Real weather only available through {weather_end.isoformat()}; held at that last "
            f"reading for the remaining ~{n_ticks - ticks_with_real_weather} of {n_ticks} ticks.",
            "Real perimeter is a single 'as of 2025-01-21' dissolved snapshot, not a timestamped "
            "progression -- no sub-14-day-granularity public perimeter data found for this fire.",
            "The unsuppressed sim self-extinguishes (runs out of reachable Fuel cells) well before "
            "n_ticks is reached and holds a stable final state after that -- check the per-500-tick "
            "progress log's state_counts for the actual saturation point on any given run.",
            "The real perimeter's area after clipping/rasterizing onto this project's study-area grid "
            "is smaller than its true unclipped area in the source dataset (see fetch_perimeter.py's "
            "own sanity print, ~23-24k acres, close to the documented ~23,400-acre real final size) -- "
            "part of the real fire's extent falls outside this project's grid bounding box, so "
            "recall/area_ratio here are against a real perimeter itself truncated by the study area.",
        ],
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full report to: {REPORT_PATH}")

    plot_overlay(sim_mask, real_mask, iou, area_ratio, recall)
    return report


if __name__ == "__main__":
    main()
