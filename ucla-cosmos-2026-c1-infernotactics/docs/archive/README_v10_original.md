InfernoTactics v10: Fire-Relative Multi-Dispatch RL
====================================================

A reinforcement-learning agent that learns to suppress wildfires in the
Palisades Fire study area (316x595 grid @ 30m resolution, 32 macro-zones,
8 real LAFD stations, 15 resource units across 4 types).

The current pipeline (v10) uses:

  - Fire-relative semantic action space
    (`active_fire`, `downwind_fire_front`, `adjacent_fuel`,
     `threatened_population`, `nearest_reachable_fire`, `noop`)
  - Policy-decided list-only multi-dispatch per simulation tick
  - Synthetic deterministic traffic model with BPR-style congestion
  - Configurable per-resource response delays
    (preparation, arrival setup, post-effect busy/reload)

This document describes only the active pipeline. For historical context,
see `context.txt`. For integration details, see `integration-instructions.txt`.


1. Requirements
---------------

  - Python 3.13+
  - PyTorch (CPU build is sufficient; the model is ~250K parameters)
  - `cosmos-venv` conda environment (or equivalent with the dependencies
    listed in `infernotactics/requirements.txt`)
  - Runtime data files in `infernotactics/data/`:
      roads.graphml, buildings.geojson, elevation.tif,
      population.tif, palisades_weather_jan2025.csv,
      grid_static.npy, grid_meta.json


2. Project Layout
-----------------

    infernotactics/
        data/                          # rebuilt runtime grid assets
        src/
            data_pipeline/              # fetch_* and config.py
            env/                        # inferno_env.py, fire_sim.py, grid_builder.py
                                        # tests: test_inferno_env.py, test_synthetic_traffic.py,
                                        #        test_multi_dispatch.py
            models/                     # relative_model.py, cnn_branch.py, mlp_branch.py,
                                        # classification_head.py, actor_critic.py
            train/                      # train_relative.py, eval_relative.py,
                                        # run_logger.py, plot_training.py,
                                        # heuristic_policy.py, relative_actions.py,
                                        # test_relative_actions.py
            validation/                 # wfigs_perimeter_validation.py
        notebooks/                      # v10_relative_actions.ipynb
        integration-instructions.txt    # integration guide for external software
        context.txt                      # historical workspace log
        models/                         # active run checkpoints (one dir per run tag)
        logs/runs/<run_tag>/             # per-run artifacts (config.json, CSVs, JSONL)
        reports/<run_tag>/               # matplotlib dashboard PNG and summary.json


3. Quick Start
--------------

Set the project root on PYTHONPATH and run:

    # Linux / macOS
    export PYTHONPATH=A:\AI\InfernoTactics\infernotactics\src

    # Windows PowerShell
    $env:PYTHONPATH = "A:\AI\InfernoTactics\infernotactics\src"

    conda run -n cosmos-venv python -m src.train.train_relative

Default configuration:

  - 100 episodes
  - synthetic traffic
  - configurable resource delays
  - list-only multi-dispatch (MAX_DISPATCH_SLOTS = 10)
  - checkpoint every 20 episodes
  - evaluation every 20 episodes on anchor, Mandeville, Getty
  - trace every 20 episodes

Override with environment variables:

    INFERNO_N_EPISODES=500
    INFERNO_RUN_TAG=my_run
    INFERNO_V8_EVAL_EVERY=50
    INFERNO_V8_CHECKPOINT_EVERY=10
    INFERNO_MAX_DISPATCH_SLOTS=5
    INFERNO_TRACE_EVERY=10


4. Inference
------------

Run the standalone inference runner:

    conda run -n cosmos-venv python -m src.train.eval_relative ^
        --checkpoint models/checkpoints_<run_tag>/latest.pt ^
        --random-points 30 ^
        --episodes 1

Flags:

    --checkpoint        path to model .pt file (required)
    --episodes          number of deterministic rollouts per ignition (default 5)
    --random-points N   use N randomly-sampled WUI ignition points instead
                        of named anchor / mandeville / getty scenarios


5. Logging and Dashboards
--------------------------

Each training run writes crash-safe artifacts to:

    logs/runs/<run_tag>/
        config.json
        train_episode.csv
        train_tick.jsonl
        eval.csv
        checkpoints.csv

When `INFERNO_TRACE_EVERY=N` is set, every Nth episode emits a
per-tick JSONL trace under `train_tick.jsonl`.

TensorBoard output is written to:

    logs/tensorboard/<run_tag>/

Start it with:

    conda run -n cosmos-venv tensorboard --logdir infernotactics/logs/tensorboard

Generate a matplotlib dashboard from the CSV logs:

    conda run -n cosmos-venv python -m src.train.plot_training ^
        --run-tag <run_tag>

Output goes to:

    reports/<run_tag>/training_dashboard.png
    reports/<run_tag>/summary.json


6. Testing
-----------

Run the active test suite:

    conda run -n cosmos-venv python -m unittest ^
        src.env.test_multi_dispatch ^
        src.env.test_synthetic_traffic ^
        src.env.test_inferno_env ^
        src.train.test_relative_actions


7. Action Interface
--------------------

The policy emits a list of `(resource_type, target_zone)` tuples per
simulation tick. Each action means "dispatch this unit from its home
station to that zone."

    action = ("helicopter", zone_id)        # dispatch a single unit
    actions = []                            # no-dispatch tick
    actions = [...]                         # multi-dispatch tick

Resource types (fixed order, indexed in `RESOURCE_TYPES`):

    0 = water_team       (suppresses active fire)
    1 = trench_crew      (creates fire breaks on fuel cells)
    2 = rescue_vehicle   (evacuates threatened buildings)
    3 = helicopter       (suppresses active fire, fast air routing)

Zone ids are integers in [0, env.n_zones). For the current build the grid
is 316x595 cells partitioned into 32 macro-zones (4 rows x 8 columns,
80 cells per side at 30m). The zone grid is defined by
`InfernoEnv._build_zones()` and is the same one the model was trained on.


8. Environment
--------------

    InfernoEnv(
        seed=None,
        traffic_mode="synthetic",     # or "legacy" for original
                                      # no-delay / no-traffic behavior
        delay_config=None,            # overrides RESOURCE_DELAY_CONFIG
                                      # default when non-None
    )

The default environment is "synthetic": deterministic BPR-style congestion
based on road class + simulation time + currently traveling ground units.
Ground resources add to road traffic; helicopters do not.

Switch to "legacy" to recover the pre-v9 behavior for direct comparison
with old logs and old checkpoints.


9. Observation Schema
---------------------

    obs = {
        "grid": np.ndarray shape (9, 316, 595) float32
                channels: 8 static layers + 1 fire-state channel
        "scalars": OrderedDict with 11 keys in this fixed order:
                    wind_speed_mph
                    wind_direction_deg
                    humidity_pct
                    water_team_available
                    trench_crew_available
                    rescue_vehicle_available
                    helicopter_available
                    time_elapsed_ticks
                    traffic_mean_load
                    traffic_max_load
                    active_ground_resources
    }


10. Roster Limits (Current Build)
----------------------------------

Real LAFD depot roster from `data_pipeline/real_depots.json`:

    water_team       3 units  (stations 69, 19, 37)
    trench_crew      4 units  (stations 19, 23, 99, 109)
    rescue_vehicle   3 units  (stations 23, 37, 71)
    helicopter       5 units  (station 114 Air Operations only)

These are the rosters used in training. Changing them in
`real_depots.json` after training invalidates the policy.


11. Versioning
--------------

The active pipeline is v10: relative actions, list-only multi-dispatch,
synthetic traffic, configurable delays.

Older artifacts (v1-v9) are no longer shipped or supported. References
to absolute-zone, zonehead, or PPO experiments have been removed from
this README and from the active code paths.


12. Integration
---------------

For embedding the policy in another simulation, see
`integration-instructions.txt`. It covers:

  - Quick-reference CLI for running inference
  - Programmatic Python harness for embedding the policy
  - The list-only action interface
  - Observation schema
  - Step output schema
  - Environment construction
  - Roster limits
  - Determinism notes
  - Logging artifact locations
  - Known limitations
  - Integration steps for a larger simulation
  - GPU/device notes
  - Versioning history
