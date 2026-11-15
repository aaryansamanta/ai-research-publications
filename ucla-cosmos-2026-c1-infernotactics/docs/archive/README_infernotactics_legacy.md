# InfernoTactics

Reinforcement-learning wildfire containment sim over the real Palisades Fire
(Jan 2025) study area: real terrain, buildings, roads, population, and
weather data feed a CNN+MLP actor-critic that dispatches fire resources
(water teams, trench crews, rescue vehicles, helicopters) across the burning
grid.

This README covers what a fresh clone needs to train a new model from
scratch. It does not cover the research history/findings — see `logs/` and
the diagnostic scripts in `src/train/` for that.

## Why data setup is a separate step

`data/*.tif`, `*.geojson`, `*.graphml`, `*.npy`, and `*.png` are gitignored
(see `.gitignore`) — they're either large binaries or straightforward to
regenerate from public sources, so they aren't committed. `models/` (model
checkpoints) **is** committed, so warm-starting from a prior run works
immediately after cloning with no extra setup.

## 1. Environment

Requires Python 3.14 (this is what the project was developed and pickled
against; earlier versions may still work but are untested).

```bash
cd infernotactics
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Fetch real-world data

Each script pulls from a live public source (OSM, USGS 3DEP, WorldPop, LA
GeoHub, NOAA ASOS) — see the docstring at the top of each file for the exact
source and any sourcing caveats. Run from `infernotactics/`:

```bash
python -m src.data_pipeline.fetch_elevation    # -> data/elevation.tif
python -m src.data_pipeline.fetch_population   # -> data/population.tif
python -m src.data_pipeline.fetch_buildings    # -> data/buildings.geojson
python -m src.data_pipeline.fetch_roads        # -> data/roads.graphml
python -m src.data_pipeline.fetch_weather      # -> data/palisades_weather_jan2025.csv (already committed, but safe to re-fetch)
```

`fetch_perimeter.py` (real fire perimeter, used only by validation, not
training) is optional at this stage:

```bash
python -m src.data_pipeline.fetch_perimeter    # -> data/palisades_perimeter_*.geojson
```

## 3. Build the simulation grid

Rasterizes everything from step 2 into the aligned static layer stack the
env and CNN branch consume:

```bash
python -m src.env.grid_builder    # -> data/grid_static.npy, data/grid_meta.json
```

## 4. Train

`src/train/` contains the full experiment history (each script is a
self-contained, one-off run — see individual docstrings). The current/final
line is `train_zonehead_randign.py` (randomized-ignition training,
warm-started from the best prior checkpoint, already present in
`models/checkpoints_zonehead_zonehead_fix1_2k/` from the clone):

```bash
INFERNO_RUN_TAG=<your_run_name> INFERNO_N_EPISODES=2000 python -m src.train.train_zonehead_randign
```

`INFERNO_RUN_TAG` is required (keeps each run's checkpoints/logs in their
own `models/checkpoints_zonehead_<tag>/` dir instead of colliding). Optional
env vars: `INFERNO_N_EPISODES` (default 2000), `INFERNO_CHECKPOINT_EVERY`
(default 25), `INFERNO_EVAL_EVERY` (default 50), `INFERNO_WARM_START_CKPT`
(default: the zonehead_fix1_2k checkpoint referenced above).

The run is crash-safe/resumable: re-running the same command with the same
`INFERNO_RUN_TAG` picks up from `resume_state.pt` in that run's checkpoint
dir if it exists.

To train a model from scratch (no warm start), use
`src/train/train_actor_critic.py` instead — the earliest, no-warm-start
entry point in this line.
