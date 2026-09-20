# InfernoTactics — Complete Tutorial

Wildfire suppression RL agent trained on real LA Westside geography and the Palisades Fire (Jan 2025).

---

## 1. Quick Start

### Prerequisites
- Windows/macOS/Linux with **Conda**
- ~8 GB free disk space (for DEM, population raster, OSM roads, building footprints)

### One-time setup
```powershell
# Clone or extract project
cd A:\AI\InfernoTactics\infernotactics

# Create and activate environment
conda create -n cosmos-venv python=3.11 -y
conda activate cosmos-venv

# Install dependencies
pip install -r requirements.txt
# If PyTorch fails, install CPU version first:
# conda install pytorch cpuonly -c pytorch

# Verify
python -c "import torch,osmnx,geopandas,rasterio; print('OK:', torch.__version__)"
```

---

## 2. Data Pipeline (Run Once)

The pipeline downloads raw data and builds the unified 30 m grid (`data/grid_static.npy`).

```powershell
cd A:\AI\InfernoTactics\infernotactics
conda activate cosmos-venv

# 1. Roads (OSM) → data/roads.graphml
python -m src.data_pipeline.fetch_roads

# 2. Building footprints (LA GeoHub) → data/buildings.geojson
python -m src.data_pipeline.fetch_buildings

# 3. Elevation (USGS 3DEP 10 m DEM) → data/elevation.tif
python -m src.data_pipeline.fetch_elevation

# 4. Population (WorldPop 1 km) → data/population.tif
python -m src.data_pipeline.fetch_population

# 5. Weather (NOAA ASOS KSMO Jan 7–8 2025) → data/palisades_weather_jan2025.csv
python -m src.data_pipeline.fetch_weather

# 6. Build unified grid → data/grid_static.npy (8, 316, 595) + data/grid_meta.json
python -m src.env.grid_builder
```

**Expected output**: `grid_static.npy` with 8 layers:
```
0: elevation        1: slope
2: building_density 3: building_height
4: road_mask        5: fuel_density (heuristic)
6: water_mask       7: population_density
```

---

## 3. Environment Sanity Check

```powershell
conda activate cosmos-venv
python -c "
from src.env.inferno_env import InfernoEnv
env = InfernoEnv(seed=42)
obs = env.reset()
print('Grid shape:', obs['grid'].shape)  # (9, 316, 595) = 8 static + 1 fire state
print('Zones:', env.n_zones)             # 32
print('Resources:', {k: len(v) for k,v in env.resources.items()})
"
```

---

## 4. Training — Three Modes

All trainers use `cosmos-venv` and read `INFERNO_RUN_TAG` for isolated checkpoints/logs.

### A. Single-Ignition (Original, 2000 episodes)
Trains on the **real Palisades Fire origin** (Skull Rock trailhead).

```powershell
$env:PYTHONPATH="A:\AI\InfernoTactics\infernotactics\src"
$env:INFERNO_N_EPISODES="2000"
$env:INFERNO_RUN_TAG="single_2000"
conda run -n cosmos-venv python -m src.train.train_actor_critic
```

- Checkpoints: `models/checkpoints_single_2000/episode_XXXX.pt`
- Logs: `logs/train_log.csv`, `logs/eval_log.csv`
- Evaluation every 20 episodes on: Skull Rock + 2 held-out (Mandeville, Getty)

### B. Multi-Ignition Curriculum (Fixed 4 scenarios, 3500 episodes)
Cycles through Skull Rock + 3 additional WUI ignitions (Topanga, Sullivan, Stone Canyon).

```powershell
$env:PYTHONPATH="A:\AI\InfernoTactics\infernotactics\src"
$env:INFERNO_N_EPISODES="3500"
$env:INFERNO_RUN_TAG="multi_3500"
conda run -n cosmos-venv python -m src.train.train_actor_critic_multi
```

### C. **v8 Relative-Action** (Recommended for generalization)
Uses **semantic targets** (`active_fire`, `adjacent_fuel`, `threatened_population`, `downwind_fire_front`, `nearest_reachable_fire`, `noop`) resolved to current zones each tick. Training samples from ~94k WUI ignition points.

```powershell
$env:PYTHONPATH="A:\AI\InfernoTactics\infernotactics\src"
$env:INFERNO_N_EPISODES="500"
$env:INFERNO_V8_EVAL_EVERY="20"
$env:INFERNO_RUN_TAG="relative_v8_500"
$env:INFERNO_V8_CHECKPOINT_EVERY="20"
conda run -n cosmos-venv python -m src.train.train_relative
```

**Key env vars:**
| Variable | Default | Purpose |
|----------|---------|---------|
| `INFERNO_N_EPISODES` | 2000 | Training length |
| `INFERNO_RUN_TAG` | `""` | Suffix for checkpoint/log dirs |
| `INFERNO_V8_EVAL_EVERY` | 50 | Evaluation cadence (v8 only) |
| `INFERNO_V8_CHECKPOINT_EVERY` | 2 | Checkpoint cadence (v8 only) |
| `INFERNO_V8_LR` | 3e-4 | Learning rate (v8 only) |
| `INFERNO_V8_AUX_COEFF` | 0.05 | Auxiliary target loss coeff (v8 only) |

**Outputs:**
- Checkpoints: `models/checkpoints_<RUN_TAG>/episode_XXXX.pt` + `latest.pt`
- Logs: `logs/train_log_<RUN_TAG>.csv`, `logs/eval_log_<RUN_TAG>.csv`
- Probe: `logs/probe_log_<RUN_TAG>.csv` (zone-logit vs fire correlation)

---

## 5. Evaluation

### Deterministic eval on named scenarios
```powershell
$env:PYTHONPATH="A:\AI\InfernoTactics\infernotactics\src"
conda run -n cosmos-venv python -m src.train.eval_relative `
  --checkpoint "models/checkpoints_relative_v8_500/latest.pt" `
  --episodes 5
```

### Randomized WUI ignition test
```powershell
conda run -n cosmos-venv python -m src.train.eval_relative `
  --checkpoint "models/checkpoints_relative_v8_500/latest.pt" `
  --random-points 30 `
  --episodes 3
```
- Samples 30 fresh ignitions from WUI pool (excludes 30-cell buffer around held-out points)
- Reports per-point: reward, buildings destroyed, containment, dispatched resources, max concurrent

### Heuristic baseline (rule-based, no learning)
```powershell
conda run -n cosmos-venv python -m src.train.heuristic_policy
```
Runs on all 4 scenarios (Skull Rock, Mandeville, Getty, Multi-ignition).

---

## 6. Architecture Summary

```
src/
├── env/
│   ├── inferno_env.py      # Gym-style env, 32 zones, real LAFD roster
│   ├── fire_sim.py         # Cellular automaton (deterministic)
│   └── grid_builder.py     # Builds 30 m aligned grid
├── models/
│   ├── cnn_branch.py       # Stride-1 convs + per-zone pooling
│   ├── mlp_branch.py       # Scalars → 128-dim
│   ├── actor_critic.py     # Factored actor (resource × zone)
│   ├── relative_model.py   # v8: resource → semantic target
│   └── inferno_model.py    # Fusion + classification head
├── train/
│   ├── train_actor_critic.py     # Single-ignition trainer
│   ├── train_relative.py         # v8 relative-action trainer
│   ├── eval.py / eval_relative.py
│   ├── heuristic_policy.py       # Rule-based baseline
│   ├── relative_actions.py       # Semantic target resolver
│   └── test_relative_actions.py  # Unit tests
└── data_pipeline/
    ├── fetch_*.py          # Raw data downloaders
    └── config.py           # BBox, paths, constants
```

---

## 7. Action Spaces

### Legacy (Absolute Zone)
```python
(resource_type, zone_id)  # zone_id ∈ [0, 31] — fixed geographic regions
```

### v8 Relative (Semantic Target)
```python
(resource_type, target_type)
target_type ∈ {active_fire, adjacent_fuel, threatened_population,
               downwind_fire_front, nearest_reachable_fire, noop}
```
Resolved at each tick by `resolve_relative_targets(env, obs)` → current zone ID.

**Resource counts (from real LAFD roster):**
| Resource | Units | Travel | Reload |
|----------|-------|--------|--------|
| water_team | 3 | Road | 5 ticks |
| trench_crew | 4 | Road | 5 ticks |
| rescue_vehicle | 3 | Road | 5 ticks |
| helicopter | 5 | Air (straight-line) | 12 ticks |

---

## 8. Reward Function

| Event | Reward |
|-------|--------|
| Fire extinguished (water/heli arrival) | +50 |
| Building destroyed | -100 × pop_mult (1×–4×) |
| Building destroyed but evacuated | -100 × pop_mult × (1 – rescue_reduction) |
| Resource wasted (no unit / no effect) | -10 |
| Travel time penalty | -0.02 × seconds |

---

## 9. Common Tasks

### Resume interrupted training
```powershell
# Just re-run with same RUN_TAG — trainer auto-resumes from latest.pt
$env:INFERNO_RUN_TAG="relative_v8_500"
conda run -n cosmos-venv python -m src.train.train_relative
```

### Test a specific checkpoint
```powershell
conda run -n cosmos-venv python -m src.train.eval_relative `
  --checkpoint "models/checkpoints_relative_v8_500/episode_0260.pt" `
  --episodes 5
```

### Compare policies
```powershell
# Heuristic baseline
python -m src.train.heuristic_policy

# Trained model (v8)
python -m src.train.eval_relative --checkpoint "models/checkpoints_relative_v8_500/best.pt"
```

### Monitor training live
```powershell
# Tail the CSV
Get-Content logs/train_log_relative_v8_500.csv -Wait -Tail 10
```

---

## 10. Project Structure After Cleanup

```
infernotactics/
├── TUTORIAL.md              # This file
├── requirements.txt
├── context.txt              # Full project history
├── data/                    # Generated (git-ignored large files)
│   ├── grid_static.npy
│   ├── grid_meta.json
│   ├── roads.graphml
│   ├── buildings.geojson
│   ├── elevation.tif
│   ├── population.tif
│   └── palisades_weather_jan2025.csv
├── models/
│   ├── checkpoints/                 # 2000-ep single-ignition run
│   │   ├── best.pt                  # Episode 260 (selected)
│   │   ├── latest.pt
│   │   └── episode_XXXX.pt
│   ├── checkpoints_relative_v8_500/ # v8 current experiment
│   └── best_model/                  # Best model archive
├── logs/
│   ├── train_log.csv
│   ├── eval_log.csv
│   ├── wfigs_validation_report.json
│   └── wfigs_validation_jan11_report.json
├── experiments/             # Archived training scripts (reference only)
│   ├── train_actor_critic_multi_v2.py ... v6
│   ├── train_multiscenario_oldloop*.py
│   ├── train_zonehead_*.py
│   ├── train_toggle_diff.py
│   └── diagnostic_*.py
└── src/
    ├── env/
    ├── models/
    ├── train/               # Core: train_actor_critic.py, train_relative.py, eval*.py, heuristic_policy.py
    ├── data_pipeline/
    └── validation/
```

---

## 11. Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: env` | Set `PYTHONPATH` to `.../infernotactics/src` |
| `grid_static.npy not found` | Run data pipeline (§2) |
| `scipy not installed` | `conda install -n cosmos-venv scipy -y` |
| MPS errors on Mac | Trainer falls back to CPU automatically |
| OOM on GPU | Reduce batch or use CPU (`device=cpu`) |
| `roads.graphml` missing | Re-run `fetch_roads` |
| Checkpoint load mismatch | Ensure model architecture matches (v8 uses `RelativeInfernoModel`) |

---

## 12. Key References

- `context.txt` — Complete project history, experimental results, negative findings
- `src/env/inferno_env.py` — Environment API, resources, rewards, ignition points
- `src/train/relative_actions.py` — Semantic target definitions & resolver
- `src/train/train_relative.py` — v8 training loop (Monte Carlo returns, no GAE/PPO)
- `logs/wfigs_validation_report.json` — Fire sim vs real perimeter (IoU 0.636)
- `experiments/` — Archived experiments with diagnostic details

---

## 13. Next Steps for Generalization

The v8 relative-action approach is the current best direction. Remaining open problems:

1. **Resource diversity** — Policy still over-relies on helicopters (only resource with +50 reward)
2. **Trench/rescue utility** — Their value is indirect; needs denser reward shaping
3. **Capacity limits** — Fire exceeds 8-unit fleet by tick 2–4; one-dispatch-per-tick is binding
4. **Action-space abstraction** — Consider fire-relative discrete actions (`n≈6–8`) instead of 32 zones

See `context.txt` sections "Action-Space Capacity Analysis" and "Future work — relative action space" for full analysis.