"""
Mechanics/perf harness for inferno_env.InfernoEnv -- NOT training. Takes
RANDOM actions for a handful of episodes to validate that reset/step/reward/
done behave sanely, and reports wall-clock timing so we have real numbers on
simulation speed before making any performance decisions.

    python -m src.env.test_inferno_env
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.fire_sim import BLAZE, THREAT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    GROUND_RESOURCE_TYPES,
    MULTI_IGNITION_TRAINING_SCENARIO,
    RESOURCE_COUNTS,
    RESOURCE_TYPES,
    InfernoEnv,
)

N_EPISODES = 5
DISPATCH_PROB = 0.7  # fraction of ticks the random policy attempts a dispatch (rest are noop)


def random_action(rng, n_zones):
    if rng.random() < DISPATCH_PROB:
        rtype = rng.choice(RESOURCE_TYPES)
        zone = int(rng.integers(n_zones))
        return (rtype, zone)
    return None


def run_episode(env, episode_seed, scenario="single"):
    rng = np.random.default_rng(episode_seed)
    obs = env.reset(seed=episode_seed, scenario=scenario)

    total_reward = 0.0
    rewards = []
    dispatch_attempts = 0
    dispatch_ok = 0
    dispatch_wasted_no_unit = 0
    dispatch_wasted_unreachable = 0
    resource_effect_success = 0
    resource_effect_wasted = 0
    effect_success_by_type = {rtype: 0 for rtype in RESOURCE_TYPES}
    effect_wasted_by_type = {rtype: 0 for rtype in RESOURCE_TYPES}
    fires_extinguished_events = 0
    buildings_destroyed = 0
    buildings_destroyed_evacuated = 0
    building_destruction_events = []  # population-aware penalty detail, see inferno_env._score_building_destruction
    availability_snapshots = []  # (tick, {rtype: n_available})
    tick_wall_times = []

    done = False
    t = 0
    while not done:
        action = random_action(rng, env.n_zones)
        actions = [action] if action is not None else []

        tick_t0 = time.perf_counter()
        obs, reward, done, info = env.step(actions)
        tick_wall_times.append(time.perf_counter() - tick_t0)

        total_reward += reward
        rewards.append(reward)

        for dispatch in info["dispatch"]:
            dispatch_attempts += 1
            status = dispatch["status"]
            if status == "dispatched":
                dispatch_ok += 1
            elif status == "no_unit_available":
                dispatch_wasted_no_unit += 1
            elif status == "zone_unreachable":
                dispatch_wasted_unreachable += 1

        for ev in info["resource_events"]:
            if ev["success"]:
                resource_effect_success += 1
                effect_success_by_type[ev["resource_type"]] += 1
                if ev["resource_type"] in ("water_team", "helicopter"):
                    fires_extinguished_events += 1
            else:
                resource_effect_wasted += 1
                effect_wasted_by_type[ev["resource_type"]] += 1

        buildings_destroyed += info["buildings_destroyed"]
        buildings_destroyed_evacuated += info["buildings_destroyed_evacuated"]
        building_destruction_events.extend(info["building_destruction_events"])

        available_now = {
            rtype: sum(1 for u in env.resources[rtype] if u["state"] == "available")
            for rtype in RESOURCE_TYPES
        }
        availability_snapshots.append(available_now)

        t += 1

    return {
        "ticks": t,
        "total_reward": total_reward,
        "rewards": rewards,
        "contained": info["contained"],
        "timeout": info["timeout"],
        "final_state_counts": info["state_counts"],
        "dispatch_attempts": dispatch_attempts,
        "dispatch_ok": dispatch_ok,
        "dispatch_wasted_no_unit": dispatch_wasted_no_unit,
        "dispatch_wasted_unreachable": dispatch_wasted_unreachable,
        "resource_effect_success": resource_effect_success,
        "resource_effect_wasted": resource_effect_wasted,
        "effect_success_by_type": effect_success_by_type,
        "effect_wasted_by_type": effect_wasted_by_type,
        "fires_extinguished_events": fires_extinguished_events,
        "buildings_destroyed": buildings_destroyed,
        "buildings_destroyed_evacuated": buildings_destroyed_evacuated,
        "building_destruction_events": building_destruction_events,
        "availability_snapshots": availability_snapshots,
        "tick_wall_times": tick_wall_times,
    }


def check_multi_station_dispatch(env):
    """Validates the v2 multi-station roster/routing/dispatch restructure
    (real_depots.json expanded from 4 flat depots to 6 per-station
    rosters; _prepare_routing()/_try_dispatch() rewritten for per-station
    Dijkstra trees + nearest-available-unit-across-stations). Three checks:
    (a) a station whose roster doesn't carry a given type never contributes
        a unit of that type; (b) nearest-available-across-stations
        genuinely picks the closer station's unit, not just the first one
        found; (c) total available units per type at reset() matches the
        real roster totals from real_depots.json. Returns a list of issue
        strings (empty if everything checks out)."""
    issues = []
    obs = env.reset(seed=999)  # noqa: F841 -- reset() call is the point, not its return value

    # (a) roster/unit-construction correctness: every unit of type rtype must
    # belong to a station whose real roster actually lists rtype.
    for rtype in RESOURCE_TYPES:
        carriers = set(env._stations_by_type.get(rtype, []))
        for unit in env.resources[rtype]:
            if unit["station_id"] not in carriers:
                issues.append(
                    f"(a) FAIL: a {rtype} unit is attributed to station {unit['station_id']!r}, "
                    f"which does not carry {rtype} in its real_depots.json roster"
                )
    if not any(i.startswith("(a)") for i in issues):
        print("(a) PASS: every resource unit belongs to a station whose real roster carries that type "
              "(no phantom units at stations lacking the type).")

    # (b) nearest-available-across-stations: find a type carried by 2+ stations
    # and a zone where those stations' travel times genuinely differ, then
    # confirm _try_dispatch() picks the closer one, not just the first in list order.
    checked_b = False
    for rtype in RESOURCE_TYPES:
        station_ids = env._stations_by_type.get(rtype, [])
        if len(station_ids) < 2:
            continue
        for zone_id in range(env.n_zones):
            per_station = {
                sid: env._station_travel_time_s[sid][zone_id]
                for sid in station_ids
                if math.isfinite(env._station_travel_time_s[sid][zone_id])
            }
            if len(per_station) < 2 or len(set(per_station.values())) < 2:
                continue  # need >=2 reachable stations with genuinely different travel times
            expected_station, expected_travel_s = min(per_station.items(), key=lambda kv: kv[1])

            env.reset(seed=999)  # fresh, all units idle
            result = env._try_dispatch(rtype, zone_id)
            checked_b = True
            if result["status"] != "dispatched":
                issues.append(f"(b) FAIL: expected a dispatch for {rtype}->zone {zone_id}, got {result['status']}")
            elif result["station_id"] != expected_station:
                issues.append(
                    f"(b) FAIL: {rtype}->zone {zone_id} dispatched from station {result['station_id']!r} "
                    f"(travel={result['travel_time_s']:.0f}s) but station {expected_station!r} "
                    f"(travel={expected_travel_s:.0f}s) was genuinely closer and had an idle unit"
                )
            else:
                print(f"(b) PASS: {rtype}->zone {zone_id} correctly dispatched from the nearer station "
                      f"{expected_station!r} ({expected_travel_s:.0f}s) over {len(per_station) - 1} "
                      f"farther alternative(s).")
            break
        if checked_b:
            break
    if not checked_b:
        issues.append("(b) INCONCLUSIVE: no (multi-station type, zone) pair with genuinely differing "
                       "reachable travel times was found -- could not exercise nearest-station selection.")

    # (c) total available units per type at reset() matches real_depots.json's roster totals.
    env.reset(seed=999)
    expected_totals = {"water_team": 3, "trench_crew": 4, "rescue_vehicle": 3, "helicopter": 5}
    for rtype in RESOURCE_TYPES:
        n_available = sum(1 for u in env.resources[rtype] if u["state"] == "available")
        if n_available != RESOURCE_COUNTS[rtype]:
            issues.append(f"(c) FAIL: {rtype} has {n_available} available units at reset, "
                           f"expected RESOURCE_COUNTS[{rtype!r}]={RESOURCE_COUNTS[rtype]}")
        if n_available != expected_totals[rtype]:
            issues.append(f"(c) FAIL: {rtype} has {n_available} available units at reset, "
                           f"expected the sanity-checked real total {expected_totals[rtype]}")
    if not any(i.startswith("(c)") for i in issues):
        print(f"(c) PASS: available units at reset() match real_depots.json roster totals: "
              f"{ {rtype: RESOURCE_COUNTS[rtype] for rtype in RESOURCE_TYPES} }")

    return issues


def check_resources_recover(availability_snapshots):
    """Did every resource type return to full availability at least once
    after having been reduced below full?"""
    ever_reduced = {rtype: False for rtype in RESOURCE_TYPES}
    ever_recovered_after_reduction = {rtype: False for rtype in RESOURCE_TYPES}
    for snap in availability_snapshots:
        for rtype in RESOURCE_TYPES:
            full = RESOURCE_COUNTS[rtype]
            if snap[rtype] < full:
                ever_reduced[rtype] = True
            elif snap[rtype] == full and ever_reduced[rtype]:
                ever_recovered_after_reduction[rtype] = True
    return ever_reduced, ever_recovered_after_reduction


def main():
    print("Building InfernoEnv (loads grid + road routing graph once, shared across all episodes)...")
    t0 = time.perf_counter()
    env = InfernoEnv(seed=123)
    env_init_s = time.perf_counter() - t0
    print(f"Env init time: {env_init_s:.2f}s")
    print(f"Zones: {env.n_zones}")
    stations_by_name = {s["station_id"]: s["station_name"] for s in env.stations}
    for rtype in RESOURCE_TYPES:
        times = env.zone_travel_time_s[rtype]
        finite = [t for t in times if math.isfinite(t)]
        station_names = ", ".join(stations_by_name[sid] for sid in env._stations_by_type.get(rtype, []))
        print(f"  {rtype:15s} stations=[{station_names}]")
        print(f"      best-case travel times (s): min={min(finite):.0f}  max={max(finite):.0f}  "
              f"unreachable={sum(1 for t in times if not math.isfinite(t))}")
    print()

    print("=== Multi-station roster/routing/dispatch checks ===")
    multi_station_issues = check_multi_station_dispatch(env)
    print()

    all_results = []
    all_reward_values = []
    all_tick_times = []

    for ep in range(N_EPISODES):
        ep_t0 = time.perf_counter()
        result = run_episode(env, episode_seed=1000 + ep)
        ep_wall_s = time.perf_counter() - ep_t0

        all_results.append(result)
        all_reward_values.extend(result["rewards"])
        all_tick_times.extend(result["tick_wall_times"])

        print(f"--- Episode {ep} ---")
        print(f"  wall time: {ep_wall_s:.2f}s over {result['ticks']} ticks "
              f"({ep_wall_s / result['ticks'] * 1000:.1f} ms/tick)")
        print(f"  terminated: contained={result['contained']}  timeout={result['timeout']}")
        print(f"  total_reward: {result['total_reward']:.1f}")
        print(f"  final state counts: {result['final_state_counts']}")
        print(f"  dispatch attempts: {result['dispatch_attempts']} "
              f"(ok={result['dispatch_ok']}, no_unit={result['dispatch_wasted_no_unit']}, "
              f"unreachable_zone={result['dispatch_wasted_unreachable']})")
        per_type = ", ".join(
            f"{rtype}: {result['effect_success_by_type'][rtype]} ok / "
            f"{result['effect_wasted_by_type'][rtype]} wasted"
            for rtype in RESOURCE_TYPES
        )
        print(f"  resource arrival effects: success={result['resource_effect_success']} "
              f"wasted={result['resource_effect_wasted']}  ({per_type})")
        print(f"  buildings destroyed: {result['buildings_destroyed']} "
              f"(evacuated first: {result['buildings_destroyed_evacuated']})")
        print()

    # --- Aggregate timing ----------------------------------------------------
    total_ticks = sum(r["ticks"] for r in all_results)
    total_wall = sum(all_tick_times)
    print("=== Timing summary ===")
    print(f"Total ticks across {N_EPISODES} episodes: {total_ticks}")
    print(f"Total step() wall time: {total_wall:.2f}s")
    print(f"Mean: {total_wall / total_ticks * 1000:.2f} ms/tick "
          f"({total_ticks / total_wall:.1f} ticks/sec)")
    print(f"Env one-time init (grid + routing graph): {env_init_s:.2f}s (amortized across all episodes)")
    print()

    # --- Sanity / anomaly checks ----------------------------------------------
    print("=== Sanity checks ===")
    issues = list(multi_station_issues)

    n_contained = sum(1 for r in all_results if r["contained"])
    print(f"Episodes fully contained by random luck: {n_contained}/{N_EPISODES}")
    n_over_max = sum(1 for r in all_results if r["ticks"] > 150)
    if n_over_max:
        issues.append(f"{n_over_max} episode(s) ran past MAX_TICKS -- done flag not firing correctly")

    distinct_rewards = len(set(round(r, 3) for r in all_reward_values))
    print(f"Distinct reward values seen across all steps: {distinct_rewards} "
          f"(of {len(all_reward_values)} steps)")
    if distinct_rewards <= 1:
        issues.append("Reward is constant regardless of action -- reward function is not wired up correctly")

    for rtype in RESOURCE_TYPES:
        recovered_any_ep = False
        for r in all_results:
            _, recovered = check_resources_recover(r["availability_snapshots"])
            if recovered[rtype]:
                recovered_any_ep = True
                break
        print(f"{rtype}: returns to full availability after being dispatched: {recovered_any_ep}")
        if not recovered_any_ep:
            issues.append(f"{rtype} never becomes available again after dispatch across any episode")

    total_dispatch_ok = sum(r["dispatch_ok"] for r in all_results)
    if total_dispatch_ok == 0:
        issues.append("No dispatch ever succeeded -- travel-time/routing or roster logic likely broken")

    print("Resource-arrival effects, aggregated across all episodes:")
    for rtype in RESOURCE_TYPES:
        ok = sum(r["effect_success_by_type"][rtype] for r in all_results)
        wasted = sum(r["effect_wasted_by_type"][rtype] for r in all_results)
        arrivals = ok + wasted
        rate = ok / arrivals if arrivals else float("nan")
        print(f"  {rtype}: {ok}/{arrivals} arrivals had an effect ({rate:.0%})")
        if arrivals >= 20 and ok == 0:
            issues.append(
                f"{rtype} never once had an effect across {arrivals} arrivals under a random policy -- "
                f"under EFFECT_RADIUS_CELLS={3}, this likely just means random zone-targeting almost never "
                f"lands the effect footprint on the (spatially small, localized) fire; expected to matter a "
                f"lot less once a trained policy can aim at the actual fire front instead of a random zone"
            )

    for rtype in RESOURCE_TYPES:
        finite_travel_times = [t for t in env.zone_travel_time_s[rtype] if math.isfinite(t)]
        mean_travel_min = (sum(finite_travel_times) / len(finite_travel_times)) / 60.0
        print(f"Mean {rtype} zone travel time from depot: {mean_travel_min:.1f} min "
              f"(range {min(finite_travel_times) / 60:.1f}-{max(finite_travel_times) / 60:.1f} min) "
              f"-- plausible for a ~{env.width * env.meta['cell_size_m'] / 1000:.0f}km x "
              f"{env.height * env.meta['cell_size_m'] / 1000:.0f}km study area")

    ground_unreachable_zones = {
        z for rtype in GROUND_RESOURCE_TYPES
        for z, t in enumerate(env.zone_travel_time_s[rtype]) if not math.isfinite(t)
    }
    if ground_unreachable_zones:
        heli_times_there = [env.zone_travel_time_s["helicopter"][z] for z in ground_unreachable_zones]
        print(f"\nZones unreachable by at least one ground resource type via the road graph: "
              f"{sorted(ground_unreachable_zones)}")
        print(f"Helicopter travel time (straight-line) to those same zones (s): "
              f"{[f'{t:.0f}' for t in heli_times_there]} -- all finite, confirming helicopter "
              f"reaches zones ground routing cannot.")
    else:
        print("\nNo zones unreachable by any ground resource type in this build of the road graph.")

    # --- Population-aware penalty check ---------------------------------------
    all_destruction_events = [ev for r in all_results for ev in r["building_destruction_events"]]
    print(f"\nBuilding-destruction events across all episodes: {len(all_destruction_events)}")
    if all_destruction_events:
        penalties = [ev["penalty_applied"] for ev in all_destruction_events]
        densities = [ev["population_density"] for ev in all_destruction_events]
        distinct_penalties = len(set(round(p, 3) for p in penalties))
        print(f"Applied penalty range: {min(penalties):.1f} to {max(penalties):.1f} "
              f"({distinct_penalties} distinct values across {len(penalties)} events) "
              f"-- population_density range seen: {min(densities):.3f} to {max(densities):.3f}")
        if distinct_penalties <= 1:
            issues.append("Every building-destroyed penalty was identical -- population scaling is not "
                           "actually varying the reward (still effectively flat -100)")
        print("Example events (sorted by population_density, low -> high):")
        for ev in sorted(all_destruction_events, key=lambda e: e["population_density"])[:2]:
            print(f"  row={ev['row']:3d} col={ev['col']:3d}  population_density={ev['population_density']:.3f}  "
                  f"multiplier={ev['multiplier']:.2f}x  evacuated={ev['evacuated']}  "
                  f"penalty_applied={ev['penalty_applied']:.1f}")
        for ev in sorted(all_destruction_events, key=lambda e: e["population_density"])[-2:]:
            print(f"  row={ev['row']:3d} col={ev['col']:3d}  population_density={ev['population_density']:.3f}  "
                  f"multiplier={ev['multiplier']:.2f}x  evacuated={ev['evacuated']}  "
                  f"penalty_applied={ev['penalty_applied']:.1f}")
    else:
        print("\nNo building-destruction events occurred across any episode -- can't verify population scaling.")

    print()
    if issues:
        print("=== FLAGGED ISSUES ===")
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print("No anomalies flagged.")

    verify_multi_ignition(env, issues)

    print()
    if issues:
        print("=== FLAGGED ISSUES (final, includes multi-ignition checks) ===")
        for issue in issues:
            print(f"  ! {issue}")


def _n_active_fire_clusters(sim):
    """Connected-component count (8-connectivity) over currently active
    (Threat/Blaze) cells -- how many separate fire fronts exist right now,
    as opposed to state_counts()'s flat cell tally."""
    from scipy.ndimage import label
    active = np.isin(sim.state, (THREAT, BLAZE))
    structure = np.ones((3, 3), dtype=bool)  # 8-connected
    _labeled, n = label(active, structure=structure)
    return n


def verify_multi_ignition(env, issues):
    """scenario='multi' -- confirm MULTI_IGNITION_TRAINING_SCENARIO actually
    starts multiple independent fire fronts that grow, and that resource
    dispatch/target-selection (which operates per-zone, see
    inferno_env._effect_target_point) still works normally with more than
    one active cluster on the grid at once."""
    print("\n=== Multi-ignition scenario check (scenario='multi') ===")
    print(f"MULTI_IGNITION_TRAINING_SCENARIO points: {MULTI_IGNITION_TRAINING_SCENARIO}")

    obs = env.reset(seed=777, scenario="multi")
    n_clusters_at_reset = _n_active_fire_clusters(env.sim)
    print(f"Active fire clusters immediately after reset: {n_clusters_at_reset} "
          f"(expected {len(MULTI_IGNITION_TRAINING_SCENARIO)}, one per ignition point)")
    if n_clusters_at_reset != len(MULTI_IGNITION_TRAINING_SCENARIO):
        issues.append(f"scenario='multi' reset produced {n_clusters_at_reset} fire clusters, "
                       f"expected {len(MULTI_IGNITION_TRAINING_SCENARIO)} (one per ignition point)")

    rng = np.random.default_rng(777)
    cluster_counts_over_time = [n_clusters_at_reset]
    active_cell_counts_over_time = [int(np.isin(env.sim.state, (THREAT, BLAZE)).sum())]
    for t in range(1, 11):
        action = random_action(rng, env.n_zones)
        obs, reward, done, info = env.step([action] if action is not None else [])
        cluster_counts_over_time.append(_n_active_fire_clusters(env.sim))
        active_cell_counts_over_time.append(info["state_counts"]["Threat"] + info["state_counts"]["Blaze"])
        if done:
            break
    print(f"Active fire clusters, tick 0->{len(cluster_counts_over_time) - 1}: {cluster_counts_over_time}")
    print(f"Active fire cell count, tick 0->{len(active_cell_counts_over_time) - 1}: {active_cell_counts_over_time}")
    if max(cluster_counts_over_time[1:], default=0) < 2:
        issues.append("scenario='multi': fewer than 2 independent fire clusters ever seen after tick 0 -- "
                       "fronts may have merged immediately or failed to grow independently")
    if active_cell_counts_over_time[-1] <= active_cell_counts_over_time[0]:
        issues.append("scenario='multi': active fire cell count did not grow over the first "
                       f"{len(active_cell_counts_over_time) - 1} ticks")

    # Run a full random-policy episode on the multi scenario (same harness as
    # run_episode()) to confirm dispatch/target-selection -- which is
    # per-zone (inferno_env._effect_target_point), not global -- still finds
    # and suppresses fire normally with 3 simultaneous fronts on the grid.
    result = run_episode(env, episode_seed=778, scenario="multi")
    print(f"\nFull scenario='multi' random-policy episode: {result['ticks']} ticks, "
          f"contained={result['contained']}, timeout={result['timeout']}, "
          f"total_reward={result['total_reward']:.1f}")
    print(f"  dispatch attempts={result['dispatch_attempts']} (ok={result['dispatch_ok']})")
    print(f"  resource arrival effects: success={result['resource_effect_success']} "
          f"wasted={result['resource_effect_wasted']}")
    print(f"  buildings destroyed: {result['buildings_destroyed']}")
    if result["dispatch_ok"] == 0:
        issues.append("scenario='multi': no dispatch ever succeeded over a full random-policy episode")
    if result["resource_effect_success"] == 0:
        issues.append("scenario='multi': no resource arrival ever had an effect over a full "
                       "random-policy episode -- target-selection may be broken with multiple fronts")

    # scenario='single' (the original default) must still work exactly as
    # before -- multi is additive, not a replacement.
    single_obs = env.reset(seed=779, scenario="single")
    single_clusters = _n_active_fire_clusters(env.sim)
    print(f"\nscenario='single' (default) still produces exactly one fire cluster: {single_clusters}")
    if single_clusters != 1:
        issues.append(f"scenario='single' produced {single_clusters} fire clusters at reset, expected 1 -- "
                       "the original single-ignition curriculum stage may have regressed")


if __name__ == "__main__":
    main()
