# InfernoTactics AI — Master Project Context

**Single consolidated source of truth.** Merges both original context documents,
de-duplicated and reconciled. Where the two originals conflicted, the *second*
(the v8/v9/v10 workspace update) wins. Verified corrections against the actual
repo are marked **[VERIFIED]** or **[CORRECTED]**.

Last consolidated: 2026-07-30.

---

## 0. THE MODEL WE ARE USING — read this first

**The canonical model is the fire-relative-action model:
`infernotactics/src/models/relative_model.py` (`RelativeInfernoModel`).**

This is v8 and everything after it (v9 synthetic traffic, v10 multi-dispatch build
directly on it). It is the deliverable — the *architecture*, not any single checkpoint of it.

| Role | File |
|---|---|
| Model | `src/models/relative_model.py` — `RelativeInfernoModel` |
| Semantic action targets | `src/train/relative_actions.py` |
| Training | `src/train/train_relative.py` |
| Evaluation | `src/train/eval_relative.py` |
| Real-time logging | `src/train/progress.py` — `EpisodeProgress` (rich Live panel + `live_status.json` tail file) |
| Action-space tests | `src/train/test_relative_actions.py` |
| Plotting (offline) | `src/train/plot_training.py` |

### Everything absolute-zone-based is history, not a candidate

`train_actor_critic.py`, `models/checkpoints/best.pt` (the +29,443 anchor result), and
the zone-head-repair run `checkpoints_zonehead_zonehead_randign_v1/` are **prior work**.
They exist to document the v1–v7 investigation. They are never the answer to "what is the
best model" — their held-out performance is −17k and −81k reward. Do not propose them.

### Why the relative action space is the whole point

Absolute zone indices (0–31) are inherently memorizable. "Dispatch to zone 18" cannot
transfer to a fire that starts anywhere else. Every architecture through v7 failed hard on
held-out ignition points for exactly this reason — and the entire v1–v7 investigation
(task-conflict fixes, credit-assignment fixes, a GAE regression bisect, a zone-head repair)
was, in retrospect, work done downstream of the real bottleneck: **the action space itself**.

The relative model never outputs an absolute zone. Each tick, the live fire state resolves
into ≤6 semantic candidates *wherever they currently are*:

- `active_fire` — zone with the most currently-burning cells
- `downwind_fire_front` — unburned fuel scored by wind alignment + distance to active fire
  (where the fire is *about to* spread)
- `adjacent_fuel` — unburned fuel bordering active fire (firebreak candidate)
- `threatened_population` — highest population density under active threat
- `nearest_reachable_fire` — active-fire zone with shortest real travel time *for the
  resource type being considered* (routing-aware, so it differs per resource type)
- `noop` — always valid; dispatch nothing this tick

The model picks a `resource_type` (4-way), then scores those candidates using each
candidate's own resolved local features (fire / fuel / population / travel-time stats) plus
a learned resource-type embedding — not a fixed zone embedding. "Send the helicopter to the
fire" means the same thing regardless of where the fire started. Side benefit: action space
shrinks 32 → ~6, so gradient signal per option goes up.

### Current best trained artifact of that architecture

**`models/checkpoints_relative_v8/latest.pt`** (episode 318+, trained with v10 multi-dispatch
under synthetic traffic + configurable delays, live logging enabled). This is the most
up-to-date model checkpoint folder. It uses the 11-scalar env interface.

Prior release: `best_model/inferno_best_model.pt` — a **self-contained release folder** (own
frozen copy of the env, grid, roads, weather, plus a verified-working `run_example.py`). It
was **episode 10 of a planned 2000-episode run** — a smoke test, deliberately shipped because
even at ep10 it demonstrates the structural win. It uses the 8-scalar interface.

### Compatibility facts you will trip over **[VERIFIED]**

- **v8 relative checkpoints expect 8 observation scalars** (`mlp.net.0.weight` = `(32, 8)`).
  **v9/v10 expect 11** (traffic scalars added). Confirmed by direct state-dict inspection.
  → v8 weights **cannot** load against the current synthetic-traffic env without a scalar
  adapter. The current `checkpoints_relative_v8/` uses the 11-scalar interface.
- **DirectML is available** via `torch-directml` (AMD Radeon 880M → `privateuseone:0`).
  Autograd support is limited; `train_relative.py` **forces CPU for training** via
  `get_device(force_cpu=True)`. Inference can use `get_device(force_cpu=False)` to pick up DirectML.
- **Checkpoint resume is automatic**: `main()` loads `checkpoints_<RUN_TAG>/latest.pt`
  and uses `INFERNO_PROGRESS_START` (or 1) as the starting episode. Re-running with the
  same `INFERNO_RUN_TAG` resumes from the last checkpoint. The checkpoint frequency is
  controlled by `CHECKPOINT_EVERY` (every N episodes), so episodes between checkpoints
  are not individually resumable — the model loads but you may repeat a few episodes.
- A training-loop bug was found and fixed before this checkpoint's later episodes: the
  gradient update recomputed the resource-type distribution **without re-masking by which
  resource types were actually available that tick**, while the rollout that generated the
  action *did* mask it. Rollout/update mismatch on the majority of ticks in any congested
  scenario (confirmed: 135 of ~171 steps in a `stone_canyon` rollout) — the same bug class
  that caused catastrophic collapse in pre-v8 architectures. Fixed in shipped code.

---

## 1. Course & project framing

UCLA ML/neural-network course final project. Must fuse 4 core ML concepts: **CNN, MLP,
Classification, RL**. Two-week timeline.

A live-interactive AI agent that learns to fight wildfires on a real model of the LA
Westside (Topanga → Brentwood/Bel-Air → Westwood/UCLA), grounded in the real **Palisades
Fire** (Jan 7–31, 2025 — ~23,400 acres, ~6,800 structures destroyed, 12 deaths).

Architecture is locked as **pure Python grid/graph simulation, NOT Unity**. The eventual
pydeck/Streamlit 3D view is a rendering layer only, built last, after training produces real
results worth showing.

```
[Real grid: fire state, fuel, elevation, roads, population] --CNN--\
                                                                    --> [Fused state] --> Actor-Critic (RL) --> Resource deployment
[Wind, humidity, resources, time] --MLP--/
```

| Concept | Role |
|---|---|
| CNN | Spatial grid: fire spread, terrain, roads, buildings, population density |
| MLP | Global scalars: wind, humidity, resource levels, time |
| Classification | Auxiliary head labeling every cell 0 Safe / 1 Fuel / 2 Threat / 3 Blaze |
| RL (Actor-Critic) | Resource deployment; critic's TD-error ties to the dopamine/RPE lecture |

### Repo layout

```
COSMOS_FINAL/
├── PROJECT_CONTEXT.md      # this file
├── context.txt             # raw original context paste (kept for provenance)
├── best_model/             # self-contained v8 relative-action release (8-scalar, ep10)
├── Simulation.html
└── infernotactics/
    ├── .venv/              # Python 3.14.3, torch 2.13.0+cpu  <-- originally the real env
    ├── data/               # real data + generated grid/sim outputs
    ├── logs/               # per-run logs + live_status.json (see training loop upgrades)
    ├── models/             # checkpoints (see §10)
    └── src/
        ├── data_pipeline/  # real data pulling (done)
        ├── env/            # grid/graph sim + fire physics + RL env (done)
        ├── models/         # CNN, MLP, classification, actor-critic, relative_model (done)
        ├── train/          # training loops + heuristic baseline + real-time logging (done)
        └── viz/            # pydeck/Streamlit (NOT started)
```

Bbox (`src/data_pipeline/config.py`): north 34.150, south 34.030, east −118.440, west −118.605.

---

## 2. Real data sources

| Data | Source | Notes |
|---|---|---|
| Roads | OSMnx / OpenStreetMap | `roads.graphml` — 3,760 nodes, 9,523 edges |
| Buildings | LA GeoHub LARIAC4 (ArcGIS FeatureServer) | `buildings.geojson` — 46,585 footprints w/ heights, capped at 150m (LiDAR outlier fix) |
| Elevation | USGS 3DEP 10m DEM (via owslib/WMS; py3dep had a Python 3.14 bug) | `elevation.tif`, 1783×948, 0–686m |
| Weather | NOAA ASOS station KSMO (Santa Monica Airport) | Real Jan 7–8 2025 hourly — humidity crashed to 0.67%, sustained wind 25.3mph: textbook Santa Ana |
| Population | US Census / ACS block-group | 9th grid layer, normalized 0–1 |
| Resource depots | Geocoded real LAFD station addresses | See §5 |
| Fire perimeter | WFIGS 2025 / LA GeoHub Palisades Perimeter | **Validation DONE** — see §7 |
| Structure damage | CAL FIRE DINS | Not yet used |

All assets were rebuilt locally from the data pipeline at one point after going missing:
`roads.graphml`, `buildings.geojson`, `elevation.tif`, `population.tif`, `grid_static.npy`,
`grid_meta.json`.

### Grid

`data/grid_static.npy` — 595×316 cells @ 30m, CRS EPSG:5070 (reprojected from
`elevation.tif`'s native CRS for pixel-perfect cross-layer alignment).

Layers: elevation, slope, building_density, building_height, road_mask, fuel_density
(*placeholder heuristic*, pending real LANDFIRE data), water_mask (coastline heuristic),
population_density (real Census).

Shape note: originally `(9, 316, 595)`; the locally rebuilt static grid is `(8, 316, 595)`
— the 9th CNN channel is live fire state, appended at observation time.

Visually verified co-registration: roads follow canyon drainages, buildings cluster in
developed areas, Topanga core empty, dense population in Westwood Village vs sparse Topanga.

---

## 3. Fire-spread simulation (`src/env/fire_sim.py`) — deterministic, NOT ML

States: Safe / Fuel / Threat / Blaze / Burned Out.
`step(wind_speed_mph, wind_direction_deg, humidity_pct)` — weather passed as live args.

Spread rule: per-Fuel-cell ignition probability = base rate × flammability × directional
slope factor × directional wind factor × road resistance × humidity suppression, combined
across 8 neighbors.

**Ember spotting:** Blaze cells have a wind-scaled chance of igniting Fuel 4–12 cells
downwind, bypassing adjacency and road resistance — can jump roads at high wind (26/30
trials at 45mph, 0/30 at 5mph). Ties to the real Palisades Fire's documented
containment-line jumps.

Validated: uphill spread ~35% wider than downhill; road corridors cut crossing 4–5×;
buildings ignite only from adjacent burning vegetation; water never ignites. A directional
bug (slope/wind vectors computed backwards) was caught via **numeric burn-centroid drift
checks**, not visual inspection, and fixed.

---

## 4. RL environment (`src/env/inferno_env.py`)

Gym-style `InfernoEnv`, `reset()` / `step()`.

**Observation:** `{'grid': 9-channel CNN input, 'scalars': MLP input via flatten_scalars()}`.
Scalars expanded **8 → 11** in v9 (`traffic_mean_load`, `traffic_max_load`,
`active_ground_resources`).

**Macro-zones:** 32 zones (80-cell blocks) — a coarser abstraction that existed purely for
the old absolute action space. The relative model does not emit zone ids, though zones still
back some of the resolved semantic candidates.

### Ignition scenarios

- `TRAINING_IGNITION_POINT` — **Skull Rock trailhead**, the real documented Palisades origin
- `VALIDATION_IGNITION_POINTS` — `mandeville_canyon`, `getty_view_park` (real,
  chaparral/WUI-adjacent, **held out**)
- `MULTI_IGNITION_TRAINING_SCENARIO` — Topanga ridge (Trippet Ranch), Sullivan Canyon
  (Brentwood), Stone Canyon (Bel-Air) — same Santa Ana corridor
- v8+ also trains from **randomized WUI ignition points**, excluding 30-cell buffers around
  both held-out validation points

**Weather:** real Jan 7–8 2025 KSMO data drives wind/humidity via `use_real_weather=True`
(default), step-hold lookup by elapsed sim time. Old fixed-45mph placeholder still available
by flag for reproducing early debug tests.

### Reward function

```
+50                                        fire extinguished
-100 x (1 + population_weight, capped)     building destroyed  (population-weighted)
-10                                        resource wasted (late arrival / already out / invalid busy-unit action)
-lambda * travel_time
-mu   * congestion_created                 (Tier 3, not implemented)
```

Bug fixed: resource effects were originally pinned to a zone's fixed geometric centroid;
now they target the actual **active-fire centroid** within the zone, falling back to
geometric centroid when there's no fire yet (correct for pre-emptive trenching).

Timing: ~122 ticks/sec; env init ~0.85s one-time (builds a Dijkstra routing tree per
station). `TICK_DURATION_MINUTES = 2.0`.

---

## 5. Resource model — overhauled, then frozen as v5

The original model was a simplification: one fixed station per resource type. A research
pass against the **LAFD Fire Station Directory** rebuilt it.

Station audit confirmed Stations 69 / 19 / 23 run real Engine / Brush-Patrol / Ambulance
apparatus — not literal "water_team / trench_crew / rescue_vehicle". **Brush Patrol (BP) is
the real analog to trench_crew.** Four more real corridor stations were added (37 Westwood/
UCLA, 71 Bel-Air, then 99 and 109 after bbox/road-graph checks). AE/ALF apparatus codes,
once flagged unconfirmed, were confirmed via the directory's own legend (Assessment Engine /
Assessment Light Force).

**Final: 8 real LAFD stations, 15 units** (up from 4 stations / 11 units):

```python
RESOURCE_COUNTS = {water_team: 3, trench_crew: 4, rescue_vehicle: 3, helicopter: 5}
```

Helicopter went 2 → 5 to match LAFD's real confirmed AW139 fleet — the single most
conservative/understated parameter in the original model.

Routing restructured: `_prepare_routing()` builds one Dijkstra tree **per station** (not per
type); `_try_dispatch()` picks the nearest available unit of a type across all stations
carrying it.

Helicopters use straight-line air routing (bypassing roads) and require a reload trip.
**1 of 32 zones is unreachable by every ground resource** via the road graph but reachable by
helicopter (280s flight) — an intentional, real air-vs-ground asymmetry.

Heuristic was re-baselined on the new env. All single-fire scenarios improved;
`multi_ignition` got ~2× worse, root-caused to a real mechanism: the heuristic's greedy
one-dispatch-per-tick rule lets the faster/more numerous trench_crew crowd out water_team and
rescue_vehicle when three fires burn at once. A heuristic tie-breaking limitation, not an
environment flaw.

**Environment frozen and signed off here.** Anything trained after this point is v5+ and is
not comparable to v1–v4.

---

## 6. v9 synthetic traffic + configurable delays

- `inferno_env.py` now defaults to `traffic_mode="synthetic"`; `traffic_mode="legacy"`
  remains for historical comparison.
- Synthetic traffic is **deterministic** — road class, simulation time, and BPR-style
  congestion from active ground-resource routes. **No external or real-time traffic
  service.**
- Ground resources (`water_team`, `trench_crew`, `rescue_vehicle`) reserve road edges while
  traveling and add configurable traffic load. **Helicopters add no road traffic.**
- `RESOURCE_DELAY_CONFIG` makes dispatch prep, arrival setup, and post-effect busy/reload
  independently configurable per resource type.
- Lifecycle in synthetic mode:
  `available → preparing → traveling → arrival_setup → effect → post_effect_busy/reloading → available`
- `src/env/test_synthetic_traffic.py`: 4 passing tests (synthetic defaults, ground route
  reservation, helicopter delay + no road load, legacy mode). Existing env-mechanics suite
  passes under synthetic mode; roster totals unchanged.

### Multi-dispatch (v10, implemented)

- Action interface is now **lists only**: `env.step([])` is a no-dispatch tick;
  `env.step([(resource_type, zone), ...])` dispatches multiple resources in one tick.
  Passing a bare tuple raises `ValueError`.
- Multi-dispatch advances resources and fire **exactly once per tick**, then processes all
  list entries — it does *not* incorrectly advance fire once per dispatch.
- The sequential decoder in `train_relative.py` uses `MAX_DISPATCH_SLOTS` (default 10) as a
  safety cap; local resource availability prevents over-selecting beyond the roster.
- `HeuristicPolicy.decide_actions()` now returns a same-tick list of greedy dispatches.
- `src/env/test_multi_dispatch.py`: 4 passing tests (list-only validation, two dispatches
  sharing one tick, empty-list advancement, same-tick availability consumption).

This work directly attacks the structural capacity ceiling in §9.

---

## 7. WFIGS perimeter validation — completed, strong positive result

Built `fetch_perimeter.py` + `wfigs_perimeter_validation.py` to numerically compare the
calibrated fire-spread sim against the real historical Palisades perimeter (LA County ArcGIS
WFIGS).

**IoU 0.636, recall 90.7%, precision 68.1%, area_ratio 1.33** — strong for an untuned
cellular automaton. Robustness-checked across 4 random seeds (stable) and against a second
independent real timestamp (Jan 11), confirming it isn't date-specific. **`fire_sim.py`
parameters were never tuned against this validation data — the number is honest, not
circular.**

Error-pattern analysis: ~52% of "missed" area sits on road cells with reduced fuel density
(explained by existing modeling choices). Over-prediction is best explained by the sim not
modeling suppression; distance, slope, and fuel were ruled out as alternatives.

This closes the old "not yet numerically validated" limitation.

---

## 8. The v1–v7 generalization investigation (prior work — keep for the writeup)

This is the scientific spine of the report. All of it was done on the **absolute** zone
action space, which is why it is prior work rather than the current model.

### The one real absolute-zone win

2000 episodes, ~2h20m, 14.36 eps/min, no crashes. Deterministic training-point reward
−96,560.8 (ep1) → +45.0 (ep2000). Entropy 1.07 → 4.81, never collapsed. Final eval:

| Scenario | RL reward | Destroyed | Containment | Heuristic | Delta |
|---|---|---|---|---|---|
| single_training | +32.1 | 0.0 | 100% | −29,410.9 | **+29,443.0** |
| mandeville_canyon | −17,489.7 | 66.7 | 0% | +30.5 | −17,520.2 |
| getty_view_park | −81,554.1 | 276.0 | 0% | +46.1 | −81,600.2 |

`best.pt` (ep260) was chosen over `latest.pt` (ep2000) — equivalent on the anchor, stronger
on both validation scenarios, and not dependent on the final episode's lucky "good mode"
landing. Headline: beats the heuristic by +29,443 on the real, severe Palisades scenario.
Honest limitation: total failure to generalize — memorization of one ignition point.

### Heuristic baseline (`src/train/heuristic_policy.py`)

Water/helicopter → nearest active-fire zone; trench → nearest fire-adjacent fuel zone with
no fire yet; rescue → zone maximizing threatened-population-density / travel-time. Uses real
road/air routing, drop-in compatible with `eval_policy()`.

| Scenario | avg_reward | destroyed | containment |
|---|---|---|---|
| single_training | −29,410.9 | 98.4 | 80% |
| mandeville_canyon | +30.5 | 0.0 | 100% |
| getty_view_park | +46.1 | 0.0 | 100% |
| multi_ignition | −59,049.6 | 228.2 | 0% |

The training scenario is genuinely harder than the validation points — consistent with being
calibrated against real severe Palisades conditions, not an easy benchmark.

### Hypothesis family 1 — task conflict (v1, v2, v3): all negative

Hypothesis: reward-scale conflict between scenarios sharing one global return normalizer and
one scalar critic (magnitudes ranged +32 to −142,000).

- **v1** multi-ignition, 3500 planned, stopped at ep1000 — no trend. One mandeville
  breakthrough (+33.1, 100%) recurred 5× between ep680–820 then vanished; one stone_canyon
  breakthrough at ep720 never recurred. Combined average oscillated in a fixed −53k to −62k
  band the whole run — bistable, not converging.
- **v2** per-scenario return normalization + scenario-identity one-hot into the MLP +
  curriculum warm-start (80/20 → 50/50 → uniform). Full 1000 episodes. Anchor held its
  solved state throughout (no catastrophic forgetting — warm-start worked); every other
  scenario landed back in the v1 bands. best.pt at ep200; 800 further episodes gained nothing.
- **v3** separate value heads per scenario (shared trunk + actor, separate critics). Heads
  verifiably diverged early (peak ~0.09 around ep40–80) then re-converged to 0.041 — the
  critics tried to specialize and couldn't sustain it. Same flat result.

Three principled fixes, all negative → **hypothesis ruled out.** Mechanism healthy
throughout (no NaN, stable entropy). A legitimate negative result, not an unresolved bug.

### Hypothesis family 2 — credit assignment (v4, v5, v6): all negative

Implemented GAE + multi-epoch PPO + per-minibatch KL clipping, unit-verified.

- Attempt 1 (no entropy floor): entropy → true zero at ep104–135, no recovery. Root cause:
  KL early-stopping has a blind spot — KL ≈ 0 between two already-near-deterministic
  distributions even as their logits diverge unboundedly.
- Attempt 2 (entropy floor): fired too early/often (ep24+), freezing the policy at 1
  minibatch/episode before real learning — no escape mechanism.
- Attempt 3 (entropy floor + gradient-spike stop): clean 25-ep dry run, then real run. Also
  caught and fixed a genuine anchor-conflict bug first (one shared actor's logits couldn't
  represent both the anchor's helicopter-favoring optimum and the broader trench/rescue
  optimum — fixed with a minimal binary anchor-identity flag).
- **v4 real run** collapsed at ep170–189, entropy near zero, no recovery, and **no detectable
  pre-collapse gradient buildup** (checked twice) — ruling out "catch it early."

Systematic sweep on the frozen 8-station env (v5):

| Lever | Result |
|---|---|
| `PPO_EPOCHS_PER_UPDATE` = 4 | collapses |
| = 2 | dips but recovers in short smoke |
| = 1 | cleanest smoke, then **still collapsed at ep170–189** in a real 500-ep run |
| grad clip 0.5 → 0.05 (10× tighter) | collapsed **faster and deeper** (ep20–29) |
| learning rate 3e-5 (10× lower) | entropy 4.42 → 0.027 by ep50, still falling; degenerate eval by ep10 |
| advantage clipping ±3σ | smooth monotonic decline to true zero by ep31 |

Advantage clipping worked *as designed* (bounding each tick's contribution gave a smooth
decline instead of a cliff) yet didn't stop the aggregate drift — early-stopping fired in 50%
of episodes vs 22% for the LR test, meaning the safety net was fighting a steady current,
not catching rare events.

**Critically: lower LR and tighter clip both collapsed *faster* than baseline.** Two
independent axes of step-size reduction producing the same inverted result ruled out "the
update step is too large" and pointed at **update direction**.

Coverage-density check: true exhaustive ignition coverage (99,657 WUI-filtered candidates)
would take ~302 hours for 1× coverage. Impractical; not pursued.

- **v6 multi-trajectory batching**: collected multiple independent episodes before each GAE
  computation and PPO update so no single trajectory could dominate. A mixing probe confirmed
  it worked (2.0–4.0 unique episodes per minibatch). **Collapsed anyway** — smoothly, fully by
  ep48, *faster* than single-trajectory baseline. KL early-stopping fired in 12/13 batches
  (92%, highest of any config). Peak RSS ~2GB, 8.38 eps/min — resources weren't the
  constraint. Directly falsified the hypothesis it was built to test, exhausting the entire
  "aggregation and step-size of the update" space.

### The bisect — regression localized to the v4 rewrite

A fact left unexploited through seven negative results: **the original 2000-episode run never
collapsed.** Every config since the v4 rewrite collapsed, *including* at
`PPO_EPOCHS_PER_UPDATE=1`, which should be near-equivalent to the old loop's single update
per episode. Two implementations that should behave alike behaved completely differently →
the seven levers had been tuning knobs on top of broken code.

Test: run the **unmodified pre-v4 loop** on the frozen v5 env, 300 episodes,
`scenario='single'`. Nothing ported, nothing modernized. Result was decisive:

- Entropy healthy all 300 episodes. Per-head: resource_entropy 1.05–1.38 of max 1.386;
  zone_entropy 3.13–3.45 of max 3.466 — above 90% of theoretical max throughout, with normal
  dip-and-recover.
- Solved the anchor: ep260/280/300 all at +42.4, 0 destroyed, 100% containment — **+29,453.3
  vs heuristic**, essentially reproducing the original headline on the new env.
- Deterministic eval converged to `('helicopter', 18)` at 100% of ticks — confident, correct,
  reached *while* training entropy stayed above 90%. The healthy signature: broad exploration
  during training, deterministic commitment at eval.
- Held-out still failed (mandeville −17,734; getty −81,621) — the generalization gap was never
  what this test checked.

**Conclusion: env and architecture both exonerated. The collapse is a regression inside the
v4 rewrite.** An apparently fundamental RL instability became a bounded, diffable code defect.

### Collapsed-policy action probe

Loaded surviving collapsed checkpoints, read out both heads' softmax across genuinely
different fire states (ticks 0/5/10/20/40):

| Run | Locked resource | Locked zone | Max prob | State-invariant |
|---|---|---|---|---|
| tight_grad_clip | rescue_vehicle | 14 | 1.000 / 1.000 | yes |
| low_lr (3e-5) | rescue_vehicle | 14 | 0.999 / 0.997 | yes |
| adv_clip | rescue_vehicle | 14 | 1.000 / 1.000 | yes |
| v6_multi_trajectory | trench_crew | 14 | — | yes |

(`v4_baseline` and the plain EPOCHS=1 run were unrecoverable — both wrote to unsuffixed
checkpoint dirs later overwritten. All runs now use `INFERNO_RUN_TAG`.)

Three of four locked onto rescue_vehicle (second-worst type), one onto trench_crew (worst),
**none onto helicopter, the only type that works.** All four locked onto **zone 14** across
different hyperparameters and runs. That shared zone is the tell: the resource heads
*disagreed*, so these weren't independent optimizations converging on a shared insight. A
run-configuration-independent attractor in the zone head means a pathology in the update
mechanism, not a policy finding a real-if-boring optimum.

### Feature-toggle diff — collapse isolated to GAE

Features re-enabled on the healthy old loop one at a time, 100 episodes each, distinct tags.

An inspection first corrected the framing: feature E (advantage normalization) was **not**
per-minibatch — v4 z-scores advantages once per episode using only that episode's ~150
values, then reuses that array across all minibatches. The old loop had no advantage
z-scoring at all; it used `normalized_return − V(s)`, where returns pass through a slow
cross-episode RunningMeanStd. So the risk is **loss of cross-episode scale**, not
small-sample minibatch noise.

| Run | Config | Result |
|---|---|---|
| 1 | E | Healthy all 100 eps, entropy >90% of max both heads |
| 2 | E+D | Real monotonic decline: resource_entropy 96% → 57%, zone_entropy healthy |
| 3 | E+D+A | **Full collapse.** Both heads exactly 0.000 by ep20, locked to (trench_crew, 17) at prob 1.000 |

Run 3 was the fastest, most total collapse of the entire investigation (every prior config
took ep104–189). **GAE is the load-bearing change**; minibatching and per-episode z-scoring
only erode slowly on their own.

### GAE isolation and the critic diagnostic — the causal explanation

GAE alone (no minibatching, no z-scoring): entropy declined to 0.12 by ep100, **but the
deterministic eval sat at the known-good solution** — +42.4, 100% containment,
`(helicopter, 18)` — at every checkpoint but one, confidence rising to 0.99. That's *benign
convergence*, categorically different from pathological lock onto wrong actions. GAE's
arithmetic is not defective; `test_gae.py`'s asserts were never going to catch this, because
the defect isn't in `compute_gae()`.

**Staleness hypothesis, tested and rejected.** Advantages are computed once per episode and
reused across ~10 sequential minibatch steps while weights move underneath. Recomputing them
fresh before each minibatch collapsed *faster* (ep15 vs Run 3's ep20), and sign agreement
with a Monte-Carlo shadow stayed at 36–70% with no trend.

**The critic diagnostic explained why.** Explained variance over 200 episodes of the old
loop, split by episode regime:

| Regime | Share | Result |
|---|---|---|
| Long / uncontained | 155/200 (77.5%) | EV +0.56 to +0.78, correlation mean **+0.875**, stable |
| Near-instantly contained | 45/200 (22.5%) | Correlation negative in **93.3%** of cases, mean **−0.78**, no improvement with training |

The critic is strong in the common case and **reliably anti-correlated in the rare one** —
and the rare regime *grows with training*: 12 of the first 100 episodes vs 33 of the last
100, as a better policy increasingly lands in its own optimum.

GAE's per-tick residual `δ_t = r_t + γV(s_{t+1}) − V(s_t)` depends on the critic tracking
value tick-to-tick. It does so reliably while fires burn — and points **the wrong way** during
exactly the short, already-winning trajectories a good policy most needs reinforced. An
aggregate explained-variance number would look net-positive and mask this completely.
Monte-Carlo returns are immune: they never need a per-tick estimate, only the final
discounted return.

**Causal chain closed: v4 rewrite → GAE → critic anti-correlated in near-terminal states →
collapse.** This also explains the inverted step-size results — reducing LR or clip removed
the noise that had been accidentally *protecting* the policy from a consistently wrong signal.

### v7 — multi-scenario on the repaired loop

1500 episodes planned on the old loop (no GAE, no minibatching, no z-scoring), 4 training
scenarios + 2 held out; stopped at ep460.

At ep100, one encouraging signal: every scenario, training and held-out alike, converged on
**helicopter** rather than scenario-specific types — consistent with the effectiveness sweep,
reducing the problem to zone selection.

But the zone column disproved the initial "bistable trade-off" reading. At every checkpoint
single_training and stone_canyon showed the *same* zone (19/19, 28/28) — **one shared global
zone preference drifting under the combined gradient of four scenarios**, not two optima
competing. stone_canyon's apparent 67% containment was collateral: it has a wide tolerance
window and benefits from whichever zone the shared logits currently favor. single_training
only works at exactly zone 18. The anchor stayed degraded for 320 consecutive episodes
(ep140–460) with no recovery; topanga_ridge and sullivan_canyon stayed flat. Curriculum
warm-start was declined — v2 had already settled that.

### Resource-type effectiveness sweep — two corrections

Fixed-policy sweep, `scenario='single'`, 5 episodes each, no learning:

| Mode | avg_reward | destroyed | containment |
|---|---|---|---|
| **helicopter-only** | **−24,068.8** | 82.0 | 80% |
| random-type | −70,904.3 | 240.6 | 40% |
| water_team-only | −115,608.0 | 390.2 | 0% |
| rescue_vehicle-only | −129,988.5 | 442.6 | 0% |
| trench_crew-only | −133,196.0 | 443.0 | 0% |

**Correction 1 — trench_crew's 73% effect-success was conditional, not intrinsic.** In
isolation it's **8%**. Mechanism: with only trench dispatching, nothing suppresses active
fire, so the front outruns the crew — targets that were "fire-adjacent, no active fire" at
dispatch have active fire by arrival, and `_apply_trench` fails on any footprint containing
Threat/Blaze. The 73% figure was measured inside a mixed policy where water/helicopter
suppression kept trench targets from going stale. This propagates back to the heuristic
re-baselining root-cause, which called trench's 73%-vs-4–8% "a genuine mechanical asymmetry"
— that framing needs the dependency noted.

**Correction 2 — helicopter dominates and mixing hurts.** Helicopter-only beats every other
single type by ~5× and beats the mixed random policy. Straight-line air routing plus the
5-unit fleet reaches fire fast enough to suppress it; the three ground types arrive too late.
Diluting dispatches across ineffective types actively costs reward. This also confirms
collapse-to-trench/rescue can't be read as finding a real optimum — it converged toward the
**worst** available choices.

### Zone-head diagnostic — the actor was ignoring fire location

A pooling-resolution test (`ADAPTIVE_POOL_SIZE` (4,4)→(8,8), 300 eps) produced no zone
differentiation: all six scenarios locked to the same zone at all 15 checkpoints. Adding
explicit per-zone features plus reward shaping also failed, freezing at (helicopter, zone 30)
for 240 consecutive episodes with bit-identical evals — **while zone_entropy stayed at
96–99.9% of max.** High entropy with state-invariant argmax is the anomaly that forced a
direct probe.

Forward passes on 20 varied observations (mixed scenarios/ticks, active fire 5 → 297 cells):

| Quantity | Result |
|---|---|
| Zone logit variance | mean 0.0039 |
| Resource logit variance (control) | mean 0.102 — **26× larger** |
| Zone logits vs real per-zone fire | r = **−0.05** flat, −0.11 per-obs |
| CNN pooled 128-d vector | variance 1.5e-7, **119/128 components exactly zero** |
| MLP output | variance 6.0 — carries weather/time, not fire location |
| Fused 256-d vector | essentially all MLP; the CNN half contributes nothing |
| `zone_pooled` per-zone features | r = **0.97** vs real per-zone fire counts |

The global average pool diluted a handful of fire cells across ~188,000 grid cells to the
point of erasure. But `zone_pooled` — same convs, different pooling — delivered excellent
zone-specific signal directly to the zone head.

**The diagnosis inverted:** not "the zone head lacks information," but "the zone head isn't
learning to use the excellent information it already receives." A learning-dynamics problem
localized to one head, not an architecture or input-resolution limit.

### Zone-head repair — first mechanically successful fix

Three changes: (1) **per-zone scorer** — each zone's logit computed from its own features
through a shared small MLP, making a constant output hard to represent; (2) **auxiliary
supervised loss** training zone logits toward the real per-zone active-fire distribution each
tick, giving dense gradient instead of episode-end only; (3) **cold-start the zone head** so
it couldn't fall back on a state-independent bias.

Clean run, ~4.5h, zero NaN, no collapse (zone_entropy 3.38 → 1.72 — specializing, not
collapsing; resource_entropy stable ~93%), aux loss 3.55 → 1.76.
**[CORRECTED]** the checkpoint dir reaches **ep1225** plus `resume_state.pt`, not 2000.

| Episode range | Flat corr | Per-obs corr |
|---|---|---|
| 25 | 0.056 | 0.072 |
| 200–500 | 0.15–0.26 | 0.20–0.32 |
| 600–1000 | 0.19–0.27 | 0.27–0.37 |
| 1500–2000 | 0.22–0.31 | 0.31–0.39 |

From −0.05 to a sustained **0.31/0.39** plateau. The zone head now genuinely reads its input.

Outcomes, mixed and honest:
- **Anchor became substantially more reliable** — solved at the majority of checkpoints from
  ep700, and critically it **loses and re-finds** the solution rather than forgetting it
  permanently (v7 lost it for good at ep140).
- **Whole-rollout behavior differentiated by scenario from ~ep500**: single_training →
  (helicopter, 18), stone_canyon → (water_team, 21). First genuine scenario-specific behavior
  in the project.
- **stone_canyon got worse** — mostly 0% containment vs its previous reliable 67%. Read as the
  fix removing a crutch: that 67% came from the shared-zone attractor, not learned behavior.
  Losing a collateral result is not a regression.

**Measurement correction:** the pre-registered test used tick-0 argmax, which is a weak test —
at tick 0 both scenarios are a ~3-cell disk in a 188,000-cell grid and are genuinely
near-identical, so identical argmax is *correct* behavior. Whole-rollout dominant action is
the valid measure, and it does differentiate.

---

## 9. Structural capacity limit — an environment ceiling, not a learning failure

`TICK_DURATION_MINUTES = 2.0`. Diagnostic on `HeuristicPolicy`, frozen v5:

| | single | stone_canyon |
|---|---|---|
| Mean episode length | 33.6 ticks | 81.4 ticks |
| Successful dispatches | 23.8 (90.3%) | 71.2 (87.7%) |
| Dispatched but no effect | 11.6/ep (49%) | 41.8/ep (59%) |
| Tick Blaze exceeds 8-unit fleet | tick 2 | tick 2–4 |
| Helicopter concurrent-active | 4.4 of 5 | 5.0 of 5 |

Busy cycles: ground types 9.5–12.4 ticks (`DEPLOYED_BUSY_TICKS=5`); helicopter 14.4–15 ticks,
of which only 2.4–3.0 is travel — **`HELICOPTER_RELOAD_TICKS=12` dominates.** Mean available
units per tick: helicopter 0.83–0.87 of 5 (83–91% utilization), trench_crew 3.33 of 4 (barely
used, consistent with its 8% isolated effect-success).

The 88–90% dispatch rate *looks* like the one-per-tick cap isn't binding, but the heuristic
self-gates and never proposes an unavailable type — that's the policy avoiding the question,
not evidence of slack.

**The binding arithmetic:** fire exceeds the 8-unit suppression fleet by tick 2–4, but
committing 8 units requires 8 ticks minimum under a one-per-tick cap, before travel. Each
helicopter is then locked ~15 ticks (~30 min sim, ~24 of it reloading), yielding roughly 27
total sorties across an 81-tick episode. The 49–59% no-effect rate is the visible symptom.

**Implication:** stone_canyon's 67% ceiling and the −121k floor are partly *capacity* results,
not purely learning results. **Some scenarios may be uncontainable by any policy — which means
they supply no successful behavior to learn from.** `HELICOPTER_RELOAD_TICKS=12` is an
unsourced modeling parameter and is the single most outcome-determining number in the env.

v10 multi-dispatch (§6) is the direct attack on this ceiling.

---

## 10. Checkpoint inventory **[VERIFIED on disk]**

| Path | What it is | Environ / Scalars | Episodes | Status |
|---|---|---|---|---|
| `models/checkpoints_relative_v8/` | **v8 relative + v10 multi-dispatch, current training run** | v10 (11 scalars, synthetic traffic, delays) | **episode 310+** (2026-07-30) | **Most up-to-date checkpoint. Auto-resumes.** Live logging enabled. `relative_v8` run tag. |
| `models/checkpoints_dml_test/` | DirectML smoke test, 1 ep | Same v10 env | ep1 | Smoke only. Has no continued training history. |
| `models/checkpoints_v11_reward_smoke/` | v11 reward-sign smoke, 3 eps | v10 env + new reward shape | ep1–3 | Smoke only. No follow-up run planned. |
| `models/checkpoints_v11_stability_check/` | v11 stability check, 4 eps | v10 env | ep2, 4 | Smoke only. |
| `models/checkpoints_relative_v10_multi_dispatch_100/` | v10 multi-dispatch, **100 episodes** | v10 (11 scalars) | 20, 40, 60, 80, 100 (2026-07-29) | **Previous longest run.** No live logging (legacy format). Post-dates best_model.pt, pre-dates checkpoints_relative_v8/. |
| `models/checkpoints_relative_v8_500/` | v8 run, reaches **ep60** + latest | 8 scalars, legacy traffic | ep1–60 | **Never evaluated, no logs.** Overlaps with the shipped release epoch. |
| `models/checkpoints_relative_v9_traffic_50/` | v9 synthetic traffic, **ep10** + latest | 11 scalars | ep10, latest | Run aborted after ep10; eval poor (anchor −112,745.5, 381.7 destroyed, 0%). |
| `models/checkpoints_relative_v9_delay_smoke/` | v9 delay smoke, ep1 | 11 scalars | ep1 | Smoke only. |
| `best_model/inferno_best_model.pt` | **Shipped release** — v8 relative, ep10 | 8 scalars, legacy traffic | ep10 | Self-contained folder (own env copy, grid, roads). Uses 8-scalar adapter. |
| `models/checkpoints_zonehead_zonehead_randign_v1/` | zone-head repair, ep1225 + resume_state | 8 scalars, prior work (absolute zones) | up to ep1225 | Prior work on the absolute action space. |
| `models/checkpoints/` | original 2000-ep run, `best.pt` = ep260 | 8 scalars, prior work (absolute zones) | up to ep2000 | Prior work; the +29,443 anchor headline. |

**The most up-to-date checkpoint folder is `checkpoints_relative_v8/`** (latest write: episode 310
at 2026-07-30 23:59, still running toward 2000). All other folders are historical or smoke runs.

**Logging gap [VERIFIED]:** All previous runs (v8, v9, v10_multi_dispatch_100) have **no**
`train_episode.csv` or `eval.csv` — their training curves are unrecoverable. **The `relative_v8`
run is the first to have full logging**, including:
- `train_episode.csv` — per-episode summary (standard CSV)
- `eval.csv` — per-checkpoint evaluation scores
- `live_status.json` — per-episode atomic JSON (tail-able)
- `train_tick.jsonl` — per-tick trace (when `TRACE_EVERY` is set)

Any pre-`relative_v8` number in the writeup must come from re-evaluating a checkpoint, not
from a log.

---

## 11. Known limitations — honest, for presentation

- **`fuel_density` is a placeholder heuristic**, not real LANDFIRE FBFM40 data.
- **Generalization gap — extensively investigated, root cause finally identified as the
  action space.** Twelve-plus configurations across three hypothesis families on the absolute
  action space: task-conflict (v1/v2/v3) all negative; credit-assignment (v4/v5/v6) all
  negative *and subsequently explained* — a bisect localized the collapse to the v4 rewrite, a
  toggle diff isolated it to GAE, and a critic diagnostic identified the mechanism (EV strong
  in long episodes, anti-correlated in near-terminal states, that regime growing as the policy
  improves); representation (pooling resolution, per-zone features, zone-head repair) raised
  zone-logit correlation from −0.05 to 0.31–0.39 and produced the first scenario-specific
  behavior, but did not close the outcome gap. **v8's fire-relative action space is what
  finally produced positive reward and full containment on both held-out points** — though only
  at a 10-episode smoke checkpoint so far, and the honest 20-point random sweep is still
  −244.6.
- **Diagnostic caveat worth recording:** the degenerate ~−119,981 / 0%-containment eval value
  was treated as corroborating evidence of collapse across several runs. The bisect showed it
  appears in *healthy* runs too (present at ep20 with entropy at 94% of max) — it is the
  reward floor for "this scenario burns uncontrolled," **not a collapse signature**. Entropy
  logs remain valid; the eval-lock timestamps cited alongside them were not independent
  confirmation.
- **Structural capacity limit** (§9) — achievable containment on the harder scenarios is
  capped by the environment, not only by the policy.
- **Helicopter fleet size**: 5 units, matching LAFD's real confirmed AW139 fleet (up from an
  understated 2).
- **Resource-to-station mapping**: now grounded in real LAFD apparatus data (8 real stations,
  real apparatus types where determinable) rather than an arbitrary single-depot-per-type
  assumption — see §5.
- **1 of 32 macro-zones is unreachable by any ground resource** — helicopter only. Intentional
  real asymmetry, not a bug.
- **Synthetic traffic is synthetic** — deterministic, road-class and BPR-based. No real-time
  traffic service is used, and this should never be described as real traffic data.
- **`HELICOPTER_RELOAD_TICKS=12` is unsourced** and is the most outcome-determining parameter
  in the environment.

---

## 11b. 3D simulation viewer — BUILT (model tied to the simulation)

`Simulation3D.html` at the repo root: a single self-contained file (~2.5 MB, no server, no
CDN, no API token) that replays the canonical relative-action policy on a real 3D model of
the basin. Built by three scripts in `infernotactics/src/viz/` — see that directory's
`README.md` for the full pipeline and regeneration commands.

```
InfernoEnv + RelativeInfernoModel --export_trajectory.py--> trajectories.json --\
                                                                                build_player.py --> Simulation3D.html
grid_static.npy --render_basemap.py--> terrain texture ------------------------/
```

**The model is what drives it.** `export_trajectory.py` runs a real deterministic rollout —
same env, weather, routing, reward and the argmax/roster-masked decode from
`eval_relative.py` — and records per-tick fire diffs, every unit's reconstructed position,
every effect with the exact cell the physics touched, every structure loss, the weather
series, the reward stream and the semantic action chosen. The browser is a renderer only;
nothing is re-simulated client-side.

**Four scenarios ship** (deterministic, seed 9100, `traffic_mode=legacy` to match the
dynamics the v8 weights were trained under):

| Scenario | Ticks | Reward | Structures lost | Contained |
|---|---|---|---|---|
| Skull Rock (training anchor) | 4 | −18.0 | 0 | yes |
| Mandeville Canyon (**held out**) | 3 | −219.7 | 0 | yes |
| Getty View Park (**held out**) | 4 | −167.2 | 0 | yes |
| Three simultaneous ignitions | 150 | −154,047.6 | 553 | **no** |

The contrast is the point: three fast air-only knockdowns, then the multi-ignition run
burning uncontrolled for the full 150-tick cap — the fleet-capacity ceiling from §9, visible.

**What the viewer shows**: real 3DEP terrain mesh (595×316 verts) textured from the real
road/building/population/water layers; fire as a per-tick GPU texture with burn scars,
scorch, flicker and firelight bleed; flame and wind-advected smoke particle systems;
helicopters flying in from Station 114 at Van Nuys (genuinely north of the grid, so they
enter off-map) with spinning rotors and water-drop bursts; ground units following real
Dijkstra routes over the OSM graph; the target macro-zone highlighted; and a HUD with the
live policy decision, ready-fleet state, Santa Ana wind compass, incident metrics, critic
V(s), scrubable timeline and event log.

**Rendering choices, all disclosed in the viewer's own provenance panel**: 3.4× vertical
exaggeration (686 m over 18 km is flat at true scale); vehicle markers symbolic in size and
distance-adaptive (a 17 m helicopter is 0.57 of a 30 m cell — sub-pixel); group dispatches
fanned into a formation because the env sends whole fleets to one zone and their
reconstructed positions coincide; and return legs animated over the post-effect busy/reload
timer, which the environment models as a duration rather than a path.

**One dynamics-neutral env change** was needed: `_advance_resources()` now records `row`/`col`
on its effect events, so the view can place drops on the cell the physics actually hit. No
change to state, reward or timing.

**Scalar adapter**: v8 checkpoints carry an 8-column MLP input while the current env emits 11
scalars. The three v9 traffic scalars are appended *last* in `SCALAR_KEYS`, so the loader
widens the layer and zero-fills them — function-identical to v8 on the eight inputs it was
trained on. Reported on stdout and in the UI.

**Verification**: shader/lighting correctness, draw counts and pixel values were checked
against the live page over the DevTools Protocol; live playback confirmed running (particles
alive, timeline advancing, log accumulating, zero exceptions). Two real bugs were found and
fixed this way — a world/view space mismatch that zeroed the terrain's diffuse term, and a
missing sRGB output encode that made the whole scene render ~20× too dark.

The legacy Cesium sketch at `Simulation.html` is left untouched for reference; it was a
decorative shell (fake 8×4 grid, hardcoded Ion token, `fetch('/get_latest_action')` against a
server that never existed) and is superseded by `Simulation3D.html`.

---

## 12. Not yet built

- Streamlit wrapper / hosted deployment (the viewer is a standalone file, opened locally)
- Tier 3 stretch: congestion penalty term in the reward (`-mu * congestion_created`),
  judge-drawn road blockages
- Real LANDFIRE fuel data
- CAL FIRE DINS structure-damage validation
- Full list-interface migration of the legacy logits adapter's remaining eval callers

---

## 13. Current next step

1. **Complete the 2000-episode training run** under tag `relative_v8` (currently at episode 310+).
   This is the active run with full logging, auto-resume, and real-time monitoring.
2. Evaluate the final checkpoint on Skull Rock, both named held-out validation points, and
   **30 random ignition points** — the random sweep is the number that matters.
3. Compare against the `checkpoints_relative_v10_multi_dispatch_100` (ep100) checkpoint to
   measure improvement from continued training.

**Cheap parallel win:** evaluate the never-measured `checkpoints_relative_v8_500/episode_0060.pt`
on the same 30-point sweep against the shipped ep10. Same env, same 8-scalar interface, no
training required. If ep60 beats −244.6, the release gets better for free.

---

## 14. Training loop upgrades (recently done)

### Real-time logging (`src/train/progress.py`)

New module providing `EpisodeProgress` — a stateful real-time training monitor.

**Terminal widget (rich Live):**
- Progress bar with ETA, episodes/sec, elapsed time
- Rolling window stats (default 50 episodes): reward (mean/min/max), destroyed, containment %, entropy, grad_norm, noop ticks
- Reward sparkline: inline bar chart of recent rewards
- Evaluation table with Δ vs previous checkpoint per scenario (green=improvement, red=regression)
- Graceful fallback to plain `print` when stdout isn't a TTY or `rich` missing

**Tail file (`live_status.json`):**
- Written atomically (`.tmp` → `os.replace`) per episode to `logs/runs/<RUN_TAG>/live_status.json`
- Tailable via `Get-Content -Wait` or `tail -f`
- Contains: rolling stats, sparkline values, eval history, last episode summary, eta, eps/sec

**Env vars:**

| Var | Default | Description |
|---|---|---|
| `INFERNO_PROGRESS_ENABLED` | `1` | Set `0` to disable rich panel |
| `INFERNO_ROLLING_WINDOW` | `50` | Episodes in rolling stats window |
| `INFERNO_PROGRESS_START` | `1` | Starting episode for resume |

### Checkpoint resume

`main()` now loads `checkpoints_<RUN_TAG>/latest.pt` if it exists. The starting episode
is set via `INFERNO_PROGRESS_START` (default 1). Re-running with the same `INFERNO_RUN_TAG`
resumes training from the last saved state (loads weights, continues learning).

### DirectML GPU support

- `get_device(force_cpu=False)` auto-detects DirectML (`privateuseone:0`) via
  `torch_directml`, or CUDA if available, then CPU as last resort.
- Training **forces CPU** (`force_cpu=True`) because DirectML's autograd doesn't support
  `torch.gather` scatter operations needed by the relative model's backward pass.
- Inference (`eval_relative.py`) can use DirectML by default.