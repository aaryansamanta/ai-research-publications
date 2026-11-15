"""Deterministic inference/evaluation for a saved v8 relative-action model."""

import argparse
import collections
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    RESOURCE_TYPES,
    TRAINING_IGNITION_POINT,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    SCALAR_KEYS,
    flatten_scalars,
)
from models.relative_model import RelativeInfernoModel  # noqa: E402
from train.relative_actions import TARGET_TYPES, decode_action  # noqa: E402
from train.train_relative import _forward  # noqa: E402
from train.train_relative import MAX_DISPATCH_SLOTS  # noqa: E402


def _select_action_argmax(logits, available):
    resource_logits = logits["resource_type"][0].clone()
    mask = torch.tensor(
        [available[rtype] > 0 for rtype in RESOURCE_TYPES],
        dtype=torch.bool,
        device=resource_logits.device,
    )
    resource_logits[~mask] = -1e9
    if not bool(mask.any()):
        return None
    r_idx = int(torch.argmax(resource_logits))
    t_idx = int(torch.argmax(logits["target"][0, r_idx]))
    return r_idx, t_idx


def eval_policy(policy, env, n_episodes=5, use_real_weather=True, deterministic=True,
               seed=0, ignition_point=None, ignition_points=None, scalars_fn=None):
    """Run any policy (model or heuristic with __call__ returning logits) against InfernoEnv.

    Returns dict with avg_reward, avg_buildings_destroyed, containment_rate, episode_rewards.
    """
    device = next(policy.parameters()).device if hasattr(policy, 'parameters') else torch.device('cpu')
    episode_rewards, episode_destroyed, episode_contained = [], [], []
    for ep in range(n_episodes):
        obs = env.reset(ignition_point=ignition_point, ignition_points=ignition_points,
                        seed=seed + ep, use_real_weather=use_real_weather)
        total_reward = 0.0
        total_destroyed = 0
        done = False
        contained = False
        while not done:
            local_available = {
                rtype: int(obs["scalars"][f"{rtype}_available"])
                for rtype in RESOURCE_TYPES
            }
            tick_actions = []
            grid_np = obs["grid"]
            scalars_np = flatten_scalars(obs["scalars"])
            grid_t = torch.from_numpy(np.ascontiguousarray(grid_np)).unsqueeze(0).to(device)
            scalars_t = torch.from_numpy(scalars_np).unsqueeze(0).to(device)
            with torch.no_grad():
                for _ in range(MAX_DISPATCH_SLOTS):
                    if hasattr(policy, '_forward') or hasattr(policy, 'forward'):
                        logits, _, _, zones = _forward(policy, obs, env, device)
                        sel = _select_action_argmax(logits, local_available)
                        if sel is None:
                            break
                        r_idx, t_idx = sel
                        action = decode_action(r_idx, t_idx, zones)
                    else:
                        action_logits, _, _ = policy(grid_t, scalars_t)
                        resource_logits = action_logits["resource_type"][0].clone()
                        mask = torch.tensor(
                            [local_available[rtype] > 0 for rtype in RESOURCE_TYPES],
                            dtype=torch.bool, device=device,
                        )
                        resource_logits[~mask] = -1e9
                        if not bool(mask.any()):
                            break
                        r_idx = int(torch.argmax(resource_logits))
                        t_idx = int(torch.argmax(action_logits["target"][0, r_idx]))
                        from train.relative_actions import resolve_relative_targets
                        zones_np, _ = resolve_relative_targets(env, obs)
                        action = decode_action(r_idx, t_idx, zones_np)
                    if action is None:
                        break
                    local_available[action[0]] -= 1
                    tick_actions.append(action)
            obs, reward, done, info = env.step(tick_actions)
            total_reward += reward
            total_destroyed += info["buildings_destroyed"]
            contained = info["contained"]
        episode_rewards.append(total_reward)
        episode_destroyed.append(total_destroyed)
        episode_contained.append(contained)
    return {
        "avg_reward": sum(episode_rewards) / n_episodes,
        "avg_buildings_destroyed": sum(episode_destroyed) / n_episodes,
        "containment_rate": sum(episode_contained) / n_episodes,
        "episode_rewards": episode_rewards,
    }


def evaluate(model, env, name, point, device, episodes, verbose=True):
    rewards, destroyed, contained, ticks = [], [], [], []
    actions = collections.Counter()
    dispatched = collections.Counter()
    max_active = collections.Counter()
    model.eval()
    with torch.no_grad():
        for episode in range(episodes):
            obs = env.reset(ignition_point=point, seed=9100 + episode, use_real_weather=True)
            total_reward = 0.0
            total_destroyed = 0
            done = False
            episode_dispatched = collections.Counter()
            episode_max_active = collections.Counter()
            while not done:
                local_available = {
                    rtype: int(obs["scalars"][f"{rtype}_available"])
                    for rtype in RESOURCE_TYPES
                }
                tick_actions = []
                for _ in range(MAX_DISPATCH_SLOTS):
                    logits, _value, _classification, target_zones = _forward(model, obs, env, device)
                    resource_logits = logits["resource_type"][0].clone()
                    available = torch.tensor(
                        [local_available[rtype] > 0 for rtype in RESOURCE_TYPES],
                        dtype=torch.bool, device=device,
                    )
                    resource_logits[~available] = -1e9
                    if not bool(available.any()):
                        break
                    resource_idx = int(torch.argmax(resource_logits))
                    target_idx = int(torch.argmax(logits["target"][0, resource_idx]))
                    action = decode_action(resource_idx, target_idx, target_zones)
                    if action is None:
                        break
                    local_available[RESOURCE_TYPES[resource_idx]] -= 1
                    tick_actions.append(action)
                    actions[(RESOURCE_TYPES[resource_idx], TARGET_TYPES[target_idx])] += 1
                obs, reward, done, info = env.step(tick_actions)
                dispatch = info.get("dispatch") or {}
                for dispatch_item in dispatch:
                    if dispatch_item.get("status") == "dispatched":
                        episode_dispatched[dispatch_item["resource_type"]] += 1
                for rtype in RESOURCE_TYPES:
                    active = sum(unit["state"] != "available" for unit in env.resources[rtype])
                    episode_max_active[rtype] = max(episode_max_active[rtype], active)
                total_reward += reward
                total_destroyed += info["buildings_destroyed"]
            rewards.append(total_reward)
            destroyed.append(total_destroyed)
            contained.append(bool(info["contained"]))
            ticks.append(info["tick"])
            dispatched.update(episode_dispatched)
            for rtype in RESOURCE_TYPES:
                max_active[rtype] = max(max_active[rtype], episode_max_active[rtype])
    result = {
        "scenario": name,
        "episodes": episodes,
        "avg_reward": float(np.mean(rewards)),
        "rewards": rewards,
        "avg_buildings_destroyed": float(np.mean(destroyed)),
        "containment_rate": float(np.mean(contained)),
        "avg_ticks": float(np.mean(ticks)),
        "semantic_actions": actions,
    }
    if verbose:
        print(f"{name}: reward={result['avg_reward']:.1f}  destroyed={result['avg_buildings_destroyed']:.1f}  "
              f"containment={result['containment_rate']:.0%}  avg_ticks={result['avg_ticks']:.1f}")
        print(f"  rewards={rewards}")
        print(f"  roster={{{', '.join(f'{r}: {len(env.resources[r])}' for r in RESOURCE_TYPES)}}}")
        print(f"  dispatched={dict(dispatched)}  max_concurrent={dict(max_active)}")
        print(f"  actions={actions}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--random-points", type=int, default=0,
                        help="Evaluate this many fresh WUI ignition points instead of named scenarios.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = InfernoEnv(seed=9100)
    obs = env.reset(seed=9100)
    model = RelativeInfernoModel(
        n_grid_channels=obs["grid"].shape[0],
        n_scalars=len(SCALAR_KEYS),
        n_resources=len(RESOURCE_TYPES),
        n_zones=env.n_zones,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"checkpoint={os.path.abspath(args.checkpoint)} device={device} episodes_per_scenario={args.episodes}")
    t_total = time.perf_counter()
    if args.random_points > 0:
        candidates = env._ignition_candidates.astype(np.float32)
        holdouts = np.array(list(VALIDATION_IGNITION_POINTS.values()), dtype=np.float32)
        distances = np.sqrt(((candidates[:, None, :] - holdouts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        candidates = candidates[distances >= 30.0]
        rng = np.random.default_rng(9317)
        count = min(args.random_points, len(candidates))
        selected = candidates[rng.choice(len(candidates), size=count, replace=False)]
        print(f"random_points: sampling {count} of {len(candidates)} candidates (>30 cells from holdouts)")
        print(f"  start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        results = []
        t_run = time.perf_counter()
        for index, point in enumerate(selected, 1):
            t_pt = time.perf_counter()
            print(f"  [{index}/{count}] igniting (row={int(point[0])}, col={int(point[1])})... ", end="", flush=True)
            res = evaluate(model, env, f"random_{index:03d}", tuple(map(int, point)), device, args.episodes, verbose=False)
            results.append(res)
            elapsed = time.perf_counter() - t_pt
            print(f"reward={res['avg_reward']:>10.1f}  destroyed={res['avg_buildings_destroyed']:>5.1f}  "
                  f"containment={res['containment_rate']:>3.0%}  ticks={res['avg_ticks']:>5.1f}  ({elapsed:.1f}s)", flush=True)
        print(f"  total: {time.perf_counter() - t_run:.1f}s  ({len(selected)} points, "
              f"{(time.perf_counter() - t_run) / max(1, len(selected)):.1f}s/point)")
        print("random aggregate: "
              f"reward={np.mean([r['avg_reward'] for r in results]):.1f}  "
              f"destroyed={np.mean([r['avg_buildings_destroyed'] for r in results]):.1f}  "
              f"containment={np.mean([r['containment_rate'] for r in results]):.0%}")
    else:
        scenarios = [("anchor", TRAINING_IGNITION_POINT)] + list(VALIDATION_IGNITION_POINTS.items())
        print(f"scenarios: {', '.join(name for name, _ in scenarios)}")
        for name, point in scenarios:
            t_pt = time.perf_counter()
            print(f"  running {name} (row={point[0]}, col={point[1]})... ", end="", flush=True)
            res = evaluate(model, env, name, point, device, args.episodes)
            print(f"  {name}: reward={res['avg_reward']:.1f}  destroyed={res['avg_buildings_destroyed']:.1f}  "
                  f"containment={res['containment_rate']:.0%}  ticks={res['avg_ticks']:.1f}  "
                  f"({time.perf_counter() - t_pt:.1f}s)", flush=True)
    print(f"done: total wall time {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
