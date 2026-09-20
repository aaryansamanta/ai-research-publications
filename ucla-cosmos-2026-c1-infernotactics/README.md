# InfernoTactics / InfernoCommand

**UCLA COSMOS 2026 · Cluster 1** — a reinforcement-learning agent that dispatches firefighting
resources against a spreading wildfire on a real model of the Los Angeles Westside, grounded in
the January 2025 Palisades Fire.

> Poster title: *InfernoCommand: A Fire-Relative Reinforcement Learning Framework for Multi-Resource
> Wildfire Response on a Real Los Angeles Terrain Model*
> Amey Garg, Indraneel Adem, Leo Lin, Ayan D. Tuteja, Aaryan Samanta

The code and the deliverables use two names for the same project: **InfernoTactics** (code, modules)
and **InfernoCommand** (poster).

## What it does

Four course concepts feed one agent:

```
 9-channel terrain / fire grid ──► CNN ──┐
                                         ├─► fused state ─► Actor-Critic (RL) ─► dispatch list
 wind, humidity, fleet status, time ─► MLP ┘        └─► auxiliary per-cell classification head
```

| Concept | Role |
|---|---|
| CNN | Spatial features from the grid (fire state, elevation, slope, buildings, roads, fuel, water, population) |
| MLP | Global scalars: wind, humidity, resource availability, elapsed time, traffic load |
| Classification | Auxiliary head labelling each cell Safe / Fuel / Threat / Blaze |
| RL (actor-critic) | Chooses which resource goes where each tick; critic TD-error ties to the reward-prediction-error lecture |

The key design choice is the **fire-relative action space**: the policy never picks an absolute zone.
Each tick it chooses among semantic targets (`active_fire`, `downwind_fire_front`, `adjacent_fuel`,
`threatened_population`, `nearest_reachable_fire`, `noop`) that resolve to wherever the fire currently
is, which is what lets it generalize to ignition points it never trained on. Full rationale, results
and history: [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md).

## Repository layout

```
├── infernotactics/          Core RL package (env, models, training, data pipeline)
│   ├── src/
│   │   ├── data_pipeline/   fetch_* scripts + config.py (bbox, data paths)
│   │   ├── env/             inferno_env.py, fire_sim.py, grid_builder.py, env tests
│   │   ├── models/          relative_model.py (canonical), cnn/mlp branches, actor_critic.py
│   │   ├── train/           train_relative.py, eval_relative.py, heuristic baseline, logging
│   │   └── validation/      Palisades perimeter validation (WFIGS)
│   ├── data/                weather CSV, grid metadata, simulation snapshots
│   ├── models/              checkpoints, one folder per run tag
│   ├── logs/                per-run CSV/JSON logs, TensorBoard events
│   ├── reports/             training dashboards + summary.json per run
│   ├── notebooks/           v10_relative_actions.ipynb
│   └── scripts/             ad-hoc check scripts (not part of the package)
├── integration/             Live Cesium 3D demo (FastAPI backend + browser client)
├── demo/                    Simulation3D.html — standalone replay viewer, no server needed
├── tools/                   camera_response_delay.py (ALERTCalifornia detection-delay prototype)
├── docs/                    poster, write-up, project context, tutorial, archived READMEs
└── archive/                 superseded code, scratch files, old logs (kept, not maintained)
```

`integration/` and `infernotactics/` must stay siblings: the servers locate the model code and
checkpoints with `../infernotactics/...`.

## Quick start

Run everything from `infernotactics/`.

```bash
cd infernotactics
python -m venv .venv && source .venv/bin/activate        # or a conda env
pip install -r requirements.txt
pip install rich tensorboard                              # imported by the code, not yet in requirements.txt
export PYTHONPATH="$PWD/src"                              # PowerShell: $env:PYTHONPATH = "$PWD\src"
```

Used with Python 3.11–3.14. CPU PyTorch is enough (~250K parameters).

**1. Rebuild the data** (large rasters are not committed; each script pulls from a public source)

```bash
python -m src.data_pipeline.fetch_elevation
python -m src.data_pipeline.fetch_population
python -m src.data_pipeline.fetch_buildings
python -m src.data_pipeline.fetch_roads
python -m src.env.grid_builder                            # -> data/grid_static.npy
```

**2. Train, evaluate, test**

```bash
python -m src.train.train_relative                        # defaults: 100 episodes, synthetic traffic
INFERNO_N_EPISODES=500 INFERNO_RUN_TAG=my_run python -m src.train.train_relative

python -m src.train.eval_relative --checkpoint models/checkpoints_relative_v10_multi_dispatch_100/latest.pt --random-points 30 --episodes 1
python -m src.train.plot_training --run-tag my_run        # -> reports/my_run/training_dashboard.png

python -m unittest src.env.test_multi_dispatch src.env.test_synthetic_traffic src.env.test_inferno_env src.train.test_relative_actions
```

More environment variables (`INFERNO_MAX_DISPATCH_SLOTS`, `INFERNO_TRACE_EVERY`, …) are listed in
[`docs/archive/README_v10_original.md`](docs/archive/README_v10_original.md), which also documents the
observation schema, action interface, and resource roster.

**3. Look at it**

- `demo/Simulation3D.html` — open in a browser. Pre-rendered replay of the trained policy on the 3D terrain.
- Live Cesium demo (needs the data from step 1 and internet access for Cesium and the camera feed):

  ```bash
  # from the repository root
  python -m uvicorn app:app --app-dir integration --host 127.0.0.1 --port 8000
  ```

  then open <http://127.0.0.1:8000>. It loads
  `infernotactics/models/checkpoints_relative_v8/latest.pt`.

## Documents

| | |
|---|---|
| [`docs/poster/`](docs/poster) | Final poster (PDF + PowerPoint source) |
| [`docs/writeup/`](docs/writeup) | Wildfire Command outline: abstract, methods, reward function |
| [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) | Master project record: data sources, every model version, results, known limitations |
| [`docs/guides/TUTORIAL.md`](docs/guides/TUTORIAL.md) | Long-form walkthrough (paths in it are from the original Windows machine) |
| [`docs/REORGANIZATION.md`](docs/REORGANIZATION.md) | Where every file moved, and what is missing from this repo |

## Honest caveats

- Traffic is **synthetic** (deterministic, road-class + BPR congestion), not real traffic data.
- `HELICOPTER_RELOAD_TICKS = 12` is unsourced and strongly affects outcomes.
- Fuel density is a placeholder heuristic pending real LANDFIRE data.
- One of the 32 macro-zones is reachable only by helicopter (real road-graph asymmetry, not a bug).

See `docs/PROJECT_CONTEXT.md` §11 (known limitations) and §12 (not yet built) for the full list.
