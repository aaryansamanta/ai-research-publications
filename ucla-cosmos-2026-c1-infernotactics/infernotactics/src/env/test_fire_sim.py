"""
Visual sanity check for fire_sim.py: ignites a single point in the hills
above the Palisades Highlands (chaparral, no buildings/roads directly on it,
close enough to development that spread reaches roads/structures within the
test window) and runs a fixed Santa-Ana-style wind (strong, offshore, from
the northeast) for a number of ticks, saving snapshot images of the fire
perimeter growing over time.

    python -m src.env.test_fire_sim
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import DATA_DIR  # noqa: E402
from env.fire_sim import (  # noqa: E402
    FireSim,
    STATE_COLORS,
    STATE_NAMES,
)

GRID_STATIC_PATH = os.path.join(DATA_DIR, "grid_static.npy")
GRID_META_PATH = os.path.join(DATA_DIR, "grid_meta.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "fire_sim_snapshots")

# Palisades Highlands hillside: high fuel_density, no building/road on the
# cell itself, ~1km from the nearest dense building cluster/road so the burn
# has room to visibly spread outward within the test window.
IGNITION_ROW, IGNITION_COL = 149, 228

N_TICKS = 80
SNAPSHOT_TICKS = [0, 10, 20, 30, 40, 55, 70, 80]

WIND_SPEED_MPH = 45.0       # strong Santa Ana
WIND_DIRECTION_DEG = 45.0   # blowing FROM the northeast, toward the southwest
HUMIDITY_PCT = 8.0          # typical Santa Ana single-digit relative humidity

CMAP = ListedColormap(STATE_COLORS)
NORM = BoundaryNorm(np.arange(len(STATE_COLORS) + 1) - 0.5, CMAP.N)


def run_and_snapshot():
    grid_static = np.load(GRID_STATIC_PATH)
    with open(GRID_META_PATH) as f:
        meta = json.load(f)

    sim = FireSim(grid_static, meta, seed=42)
    print(f"Grid: {sim.width}x{sim.height} cells @ {sim.cell_size_m}m")
    print(f"Ignition at row={IGNITION_ROW}, col={IGNITION_COL} "
          f"(elevation={sim.elevation[IGNITION_ROW, IGNITION_COL]:.0f}m)")
    print(f"Wind: {WIND_SPEED_MPH}mph from {WIND_DIRECTION_DEG} deg, "
          f"humidity={HUMIDITY_PCT}%")

    sim.ignite(IGNITION_ROW, IGNITION_COL, radius=1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snapshots = {}

    if 0 in SNAPSHOT_TICKS:
        snapshots[0] = sim.state.copy()

    for t in range(1, N_TICKS + 1):
        sim.step(
            wind_speed_mph=WIND_SPEED_MPH,
            wind_direction_deg=WIND_DIRECTION_DEG,
            humidity_pct=HUMIDITY_PCT,
        )
        if t in SNAPSHOT_TICKS:
            snapshots[t] = sim.state.copy()
        counts = sim.state_counts()
        if t % 10 == 0 or t == N_TICKS:
            print(f"tick {t:3d}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    _save_grid_figure(snapshots, sim)
    _save_individual_frames(snapshots, sim)


def _save_grid_figure(snapshots, sim):
    ticks = sorted(snapshots)
    n = len(ticks)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, t in zip(axes, ticks):
        _render(ax, snapshots[t], sim, t)
    for ax in axes[n:]:
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[i]) for i in range(len(STATE_NAMES))
    ]
    fig.legend(handles, STATE_NAMES, loc="lower center", ncol=len(STATE_NAMES))
    fig.suptitle(
        f"Fire spread: ignition ({sim.elevation[IGNITION_ROW, IGNITION_COL]:.0f}m elev), "
        f"wind {WIND_SPEED_MPH}mph @ {WIND_DIRECTION_DEG}deg (NE, Santa-Ana-style), "
        f"humidity {HUMIDITY_PCT}%",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path = os.path.join(DATA_DIR, "fire_sim_progression.png")
    fig.savefig(out_path, dpi=140)
    print(f"Saved progression grid -> {out_path}")


def _save_individual_frames(snapshots, sim):
    for t, state in snapshots.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        _render(ax, state, sim, t)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"tick_{t:03d}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
    print(f"Saved {len(snapshots)} individual frames -> {OUTPUT_DIR}/")


def _render(ax, state, sim, tick):
    # Muted terrain + buildings + roads backdrop for geographic context; only
    # actively-burning/threat/burned-out cells pop with a vivid overlay, so
    # the fire perimeter is easy to read against unburned Safe/Fuel terrain.
    ax.imshow(sim.elevation, cmap="terrain", origin="upper", alpha=0.55)

    building_overlay = np.ma.masked_where(sim.building_density <= 0, sim.building_density)
    ax.imshow(building_overlay, cmap="Greys", origin="upper", alpha=0.6, vmin=0, vmax=1)

    road_overlay = np.ma.masked_where(~sim.road_mask, sim.road_mask)
    ax.imshow(road_overlay, cmap="Blues", origin="upper", alpha=0.8, vmin=0, vmax=2)

    active = state >= 2  # Threat, Blaze, Burned Out -- hide Safe/Fuel so backdrop shows
    fire_overlay = np.ma.masked_where(~active, state)
    ax.imshow(fire_overlay, cmap=CMAP, norm=NORM, origin="upper", alpha=0.9)

    ax.scatter([IGNITION_COL], [IGNITION_ROW], marker="*", s=80, c="black", zorder=5)
    ax.set_title(f"tick {tick}")
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    run_and_snapshot()
