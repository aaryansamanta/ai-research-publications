"""V8/v10 training loop: randomized ignitions with fire-relative actions
and policy-decided list-only multi-dispatch per simulation tick.

Uses the proven pre-v4 Monte Carlo loop.  The action representation is
fire-relative (semantic targets), the dispatch count per tick is chosen
by the policy, and the environment runs in synthetic-traffic mode with
configurable per-resource delays.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inferno_env import (  # noqa: E402
    RESOURCE_TYPES,
    SCALAR_KEYS,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    TRAINING_IGNITION_POINT,
    flatten_scalars,
)
from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from models.classification_head import fire_state_to_class  # noqa: E402
from models.relative_model import RelativeInfernoModel  # noqa: E402
from train.relative_actions import (  # noqa: E402
    N_TARGET_TYPES,
    TARGET_TYPES,
    decode_action,
    resolve_relative_targets,
)
from train.run_logger import RunLogger, summarize_episode  # noqa: E402
from train.progress import EpisodeProgress  # noqa: E402


# Hyperparameters (previously in train_actor_critic.py; inlined here after that
# file was removed in the v10 reorganization to keep only the active pipeline).
CLASSIFICATION_LOSS_COEFF = 0.3
ENTROPY_COEFF = 0.02  # bumped from 0.01 to maintain exploration under v11's larger reward magnitudes
GAMMA = 0.99
GRAD_CLIP_NORM = 0.5


class RunningMeanStd:
    """Welford's online mean/variance, used to normalize returns before they
    hit the loss. epsilon-initialized count so the very first episode doesn't
    divide by zero; normalize() is called BEFORE update() each episode so an
    episode's own returns don't bias the statistics used to normalize it.

    v11 reward scale: a 100-cell fire running for 50 ticks now produces
    -50*100*50 = -250000 just from the per-tick fire penalty alone (on top of
    buildings destroyed and dispatch costs).  Old defaults of mean=0, var=1
    produced huge initial advantages and policy gradients that exploded on
    the first episode.  We seed the variance with a larger value (~5000^2)
    so early advantages are in a reasonable range and Adam's per-parameter
    learning rates can adapt.
    """

    def __init__(self, epsilon=1e-4, init_var=5e3 ** 2):
        self.mean = 0.0
        self.var = float(init_var)
        self.count = epsilon

    def update(self, values):
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        batch_count = len(values)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, values):
        return [(v - self.mean) / (math.sqrt(self.var) + 1e-8) for v in values]


def compute_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def get_device(force_cpu: bool = False):
    # Try DirectML (AMD/Intel GPUs on Windows) for inference
    try:
        import torch_directml
        if torch_directml.is_available() and not force_cpu:
            return torch_directml.device()
    except Exception:
        pass
    # Fallback: CUDA
    if torch.cuda.is_available() and not force_cpu:
        return torch.device("cuda")
    # Fallback: CPU
    return torch.device("cpu")


def _categorical_log_prob(logits, value):
    """log_prob for a Categorical without torch.gather. DirectML can't
    backprop through gather when the sampled dim sizes differ, so select
    the sampled entry with a one-hot dot product instead."""
    logp = F.log_softmax(logits, dim=-1)
    onehot = torch.zeros(logp.shape[-1], dtype=logp.dtype, device=logp.device)
    onehot[value.to(device=logp.device).long()] = 1.0
    return torch.dot(logp, onehot)


def _categorical_entropy(logits):
    """Shannon entropy of a Categorical logits vector, gather-free."""
    p = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(-1)



N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 2000))
BASE_SEED = 8200
LEARNING_RATE = float(os.environ.get("INFERNO_V8_LR", 1e-4))
AUX_TARGET_LOSS_COEFF = float(os.environ.get("INFERNO_V8_AUX_COEFF", 0.05))
STATUS_EVERY = 20
EVAL_EVERY = int(os.environ.get("INFERNO_V8_EVAL_EVERY", 50))
RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "relative_v8")
CHECKPOINT_EVERY = int(os.environ.get("INFERNO_V8_CHECKPOINT_EVERY", 2))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", f"checkpoints_{RUN_TAG}")
MAX_DISPATCH_SLOTS = int(os.environ.get("INFERNO_MAX_DISPATCH_SLOTS", 10))
TRACE_EVERY = int(os.environ.get("INFERNO_TRACE_EVERY", 0))
PROGRESS_ENABLED = os.environ.get("INFERNO_PROGRESS_ENABLED", "1") != "0"
PROGRESS_START = int(os.environ.get("INFERNO_PROGRESS_START", 1))
PROGRESS_WINDOW = int(os.environ.get("INFERNO_ROLLING_WINDOW", 50))
FORCE_CPU = os.environ.get("INFERNO_FORCE_CPU", "0") != "0"

print(f"[relative_v10] N_EPISODES={N_EPISODES} LEARNING_RATE={LEARNING_RATE} "
      f"AUX_TARGET_LOSS_COEFF={AUX_TARGET_LOSS_COEFF} STATUS_EVERY={STATUS_EVERY} EVAL_EVERY={EVAL_EVERY} "
      f"RUN_TAG={RUN_TAG} CHECKPOINT_EVERY={CHECKPOINT_EVERY} MAX_DISPATCH_SLOTS={MAX_DISPATCH_SLOTS} TRACE_EVERY={TRACE_EVERY} "
      f"PROGRESS_ENABLED={PROGRESS_ENABLED} PROGRESS_START={PROGRESS_START} PROGRESS_WINDOW={PROGRESS_WINDOW}")


def _tensor_targets(env, obs, device):
    zones, features = resolve_relative_targets(env, obs)
    return (
        torch.from_numpy(zones).unsqueeze(0).to(device),
        torch.from_numpy(features).unsqueeze(0).to(device),
        zones,
    )


def _resource_mask(obs):
    return torch.tensor([
        obs["scalars"][f"{rtype}_available"] > 0 for rtype in RESOURCE_TYPES
    ], dtype=torch.bool)


def _forward(model, obs, env, device):
    grid, scalars = RelativeInfernoModel.obs_to_tensors(obs, device=device) if hasattr(RelativeInfernoModel, "obs_to_tensors") else (
        torch.from_numpy(np.ascontiguousarray(obs["grid"])).unsqueeze(0).to(device),
        torch.from_numpy(flatten_scalars(obs["scalars"])).unsqueeze(0).to(device),
    )
    target_zones, target_features, raw_zones = _tensor_targets(env, obs, device)
    logits, value, classification = model(grid, scalars, target_zones, target_features)
    return logits, value, classification, raw_zones


def collect_rollout(env, model, ignition_point, device, seed):
    obs = env.reset(ignition_point=ignition_point, use_real_weather=True, seed=seed)
    steps, total_reward, buildings_destroyed = [], 0.0, 0
    done = False
    info = None
    with torch.no_grad():
        while not done:
            tick_obs = obs
            tick_actions = []
            local_available = {
                rtype: int(tick_obs["scalars"][f"{rtype}_available"])
                for rtype in RESOURCE_TYPES
            }
            for _ in range(MAX_DISPATCH_SLOTS):
                logits, _value, _classification, raw_zones = _forward(model, tick_obs, env, device)
                resource_logits = logits["resource_type"][0].clone()
                available = torch.tensor(
                    [local_available[rtype] > 0 for rtype in RESOURCE_TYPES],
                    dtype=torch.bool, device=device,
                )
                resource_logits[~available] = -1e9
                if not bool(available.any()):
                    break
                resource_idx = int(Categorical(logits=resource_logits).sample())
                target_idx = int(Categorical(logits=logits["target"][0, resource_idx]).sample())
                action = decode_action(resource_idx, target_idx, raw_zones)
                if action is None:
                    break
                local_available[RESOURCE_TYPES[resource_idx]] -= 1
                tick_actions.append({
                    "resource_idx": resource_idx,
                    "target_idx": target_idx,
                    "target_zones": raw_zones,
                    "resource_mask": available.detach().cpu().numpy(),
                })

            actions = [
                decode_action(a["resource_idx"], a["target_idx"], a["target_zones"])
                for a in tick_actions
            ]
            next_obs, reward, done, info = env.step([a for a in actions if a is not None])
            steps.append({
                "grid": tick_obs["grid"],
                "scalars": flatten_scalars(tick_obs["scalars"]),
                "actions": tick_actions,
                "fire_state_target": fire_state_to_class(torch.from_numpy(tick_obs["grid"][-1]).long()),
                "info": info,
                "reward": reward,
            })
            total_reward += reward
            buildings_destroyed += info["buildings_destroyed"]
            obs = next_obs
    return steps, total_reward, buildings_destroyed, info["contained"]


def _aux_target(resource_idx, target_zones):
    # Suppression and nearest-fire candidates are interchangeable operationally;
    # prefer the semantic candidate that exposes the transferable rule.
    preferred = {0: 0, 1: 2, 2: 3, 3: 0}[resource_idx]
    if target_zones[resource_idx, preferred] >= 0:
        return preferred
    valid = np.flatnonzero(target_zones[resource_idx] >= 0)
    return int(valid[0]) if len(valid) else N_TARGET_TYPES - 1


def update_policy(model, optimizer, steps, device, normalizer):
    raw_returns = compute_returns([s["reward"] for s in steps], GAMMA)
    returns = normalizer.normalize(raw_returns)
    normalizer.update(raw_returns)
    optimizer.zero_grad()
    sums = [0.0] * 7
    for step, g_t in zip(steps, returns):
        grid = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
        scalars = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
        target_zones_np = step["actions"][0]["target_zones"] if step["actions"] else np.full(
            (len(RESOURCE_TYPES), N_TARGET_TYPES), -1, dtype=np.int64
        )
        zones = torch.from_numpy(target_zones_np).unsqueeze(0).to(device)
        features_np = resolve_relative_targets_from_state(step["grid"], step["scalars"], target_zones_np)
        features = torch.from_numpy(features_np).unsqueeze(0).to(device)
        logits, value, classification = model(grid, scalars, zones, features)
        log_prob = torch.tensor(0.0, device=device)
        entropy = torch.tensor(0.0, device=device)
        target_entropy = torch.tensor(0.0, device=device)
        resource_entropy = torch.tensor(0.0, device=device)
        aux_loss = torch.tensor(0.0, device=device)
        for action_data in step["actions"]:
            resource_idx = action_data["resource_idx"]
            target_idx = action_data["target_idx"]
            resource_logits = logits["resource_type"][0].clone()
            resource_mask = torch.from_numpy(action_data["resource_mask"]).to(device)
            resource_logits[~resource_mask] = -1e9
            log_prob = log_prob + _categorical_log_prob(resource_logits, torch.tensor(resource_idx, device=device)) \
                + _categorical_log_prob(logits["target"][0, resource_idx], torch.tensor(target_idx, device=device))
            entropy = entropy + _categorical_entropy(logits["target"][0, resource_idx])
            target_entropy = target_entropy + _categorical_entropy(logits["target"][0, resource_idx])
            resource_entropy = resource_entropy + _categorical_entropy(resource_logits)
            aux_idx = _aux_target(resource_idx, target_zones_np)
            aux_loss = aux_loss + F.cross_entropy(
                logits["target"][0, resource_idx].unsqueeze(0),
                torch.tensor([aux_idx], device=device),
            )
        value_scalar = value.squeeze()
        target_return = torch.tensor(g_t, dtype=value_scalar.dtype, device=device)
        advantage = (target_return - value_scalar).detach()
        classification_loss = F.cross_entropy(classification, step["fire_state_target"].unsqueeze(0).to(device))
        policy_loss = -log_prob * advantage
        value_loss = (value_scalar - target_return) ** 2
        loss = policy_loss + 0.5 * value_loss + CLASSIFICATION_LOSS_COEFF * classification_loss + AUX_TARGET_LOSS_COEFF * aux_loss - ENTROPY_COEFF * entropy
        loss.backward()
        sums[0] += policy_loss.item(); sums[1] += value_loss.item(); sums[2] += classification_loss.item()
        sums[3] += aux_loss.item(); sums[4] += entropy.item(); sums[5] += resource_entropy.item(); sums[6] += target_entropy.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    n = len(steps)
    return tuple(x / n for x in sums)


def resolve_relative_targets_from_state(grid, scalars, target_zones):
    """Rebuild candidate features during update without retaining env objects."""
    fire = grid[-1]
    active = np.isin(fire, (2, 3))
    fuel = fire == 1
    adjacent = _dilate(active) & fuel
    building = grid[2] > 0.10
    population = grid[7]
    rows, cols = fire.shape
    out = np.zeros((len(RESOURCE_TYPES), N_TARGET_TYPES, 10), dtype=np.float32)
    for r in range(len(RESOURCE_TYPES)):
        for t, zone in enumerate(target_zones[r]):
            if zone < 0:
                continue
            # Zone indices are stable only for feature reconstruction; the
            # action itself remains relative because the resolver selected it.
            zr, zc = divmod(int(zone), 8)
            r0, r1 = zr * 80, min((zr + 1) * 80, rows)
            c0, c1 = zc * 80, min((zc + 1) * 80, cols)
            region = fire[r0:r1, c0:c1]
            threat = building[r0:r1, c0:c1] & active[r0:r1, c0:c1]
            out[r, t] = [1.0, min(float(active[r0:r1, c0:c1].sum()) / 256, 1), min(float(adjacent[r0:r1, c0:c1].sum()) / 256, 1), float(population[r0:r1, c0:c1][threat].max()) if threat.any() else 0.0, float((region == 1).mean()), float(building[r0:r1, c0:c1].mean()), 0.0, r0 / rows, c0 / cols, float(active[r0:r1, c0:c1].any())]
    return out


def _dilate(mask):
    out = np.zeros_like(mask, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out[max(0, dr):min(mask.shape[0], mask.shape[0] + dr), max(0, dc):min(mask.shape[1], mask.shape[1] + dc)] |= mask[max(0, -dr):min(mask.shape[0], mask.shape[0] - dr), max(0, -dc):min(mask.shape[1], mask.shape[1] - dc)]
    return out


def evaluate(model, env, point, device, episodes=2):
    rewards, destroyed, contained = [], [], []
    model.eval()
    with torch.no_grad():
        for ep in range(episodes):
            obs = env.reset(ignition_point=point, seed=BASE_SEED + ep, use_real_weather=True)
            total, lost, done = 0.0, 0, False
            while not done:
                logits, _v, _c, zones = _forward(model, obs, env, device)
                resource_logits = logits["resource_type"][0].clone()
                available = _resource_mask(obs).to(device)
                resource_logits[~available] = -1e9
                ri = int(torch.argmax(resource_logits)) if bool(available.any()) else 0
                ti = int(torch.argmax(logits["target"][0, ri]))
                action = decode_action(ri, ti, zones)
                obs, reward, done, info = env.step([action] if action is not None else [])
                total += reward; lost += info["buildings_destroyed"]
            rewards.append(total); destroyed.append(lost); contained.append(info["contained"])
    model.train()
    return float(np.mean(rewards)), float(np.mean(destroyed)), float(np.mean(contained))


def save_checkpoint(model, episode):
    """Save frequent lightweight model snapshots for crash recovery/inspection."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = os.path.join(CHECKPOINT_DIR, f"episode_{episode:04d}.pt")
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(model.state_dict(), checkpoint)
    torch.save(model.state_dict(), latest)
    return checkpoint


def main():
    device = get_device(force_cpu=FORCE_CPU)
    env = InfernoEnv(seed=BASE_SEED)
    obs = env.reset(seed=BASE_SEED)
    model = RelativeInfernoModel(len(obs["grid"]), len(SCALAR_KEYS), len(RESOURCE_TYPES), env.n_zones).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    normalizer = RunningMeanStd()

    # Resume from latest checkpoint if available
    start_episode = 1
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pt")
    print(f"[relative_v10] Looking for latest checkpoint at {latest_checkpoint}")
    if os.path.exists(latest_checkpoint):
        print(f"[relative_v10] Resuming from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        # Try to extract episode number from checkpoints.csv
        start_episode = PROGRESS_START if PROGRESS_START is not None else 1

    pool = env._ignition_candidates
    holdout = np.array(list(VALIDATION_IGNITION_POINTS.values()), dtype=np.float32)
    distances = np.sqrt(((pool[:, None, :].astype(np.float32) - holdout[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    pool = pool[distances >= 30.0]
    logger = RunLogger(PROJECT_ROOT, RUN_TAG, {
        "episodes": N_EPISODES, "device": str(device), "traffic_mode": env.traffic_mode,
        "delay_config": env.delay_config, "max_dispatch_slots": MAX_DISPATCH_SLOTS,
        "learning_rate": LEARNING_RATE, "gamma": GAMMA, "scalar_keys": SCALAR_KEYS,
        "resource_counts": {rtype: len(env.resources[rtype]) for rtype in RESOURCE_TYPES},
    }, trace_every=TRACE_EVERY)
    print(f"[relative_v10] device={device} episodes={N_EPISODES} ignition_pool={len(pool)} run={RUN_TAG} start_episode={start_episode}")
    rng = np.random.default_rng(BASE_SEED + 1)
    eval_scenarios = ["anchor", *VALIDATION_IGNITION_POINTS.keys()]
    progress = EpisodeProgress(
        n_episodes_start=start_episode,
        n_episodes=N_EPISODES,
        run_dir=logger.run_dir,
        run_tag=RUN_TAG,
        env_cfg={
            "device": str(device),
            "traffic_mode": env.traffic_mode,
            "max_dispatch_slots": MAX_DISPATCH_SLOTS,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
        },
        rolling_window=PROGRESS_WINDOW,
        eval_scenarios=eval_scenarios,
        enable=PROGRESS_ENABLED,
    )
    print(f"[relative_v10] progress={progress.status_file} live={progress.is_live} window={PROGRESS_WINDOW}", flush=True)
    try:
        with progress:
            for episode in range(start_episode, N_EPISODES + 1):
                point = tuple(int(x) for x in pool[rng.integers(len(pool))])
                steps, reward, destroyed, contained = collect_rollout(env, model, point, device, BASE_SEED + episode)
                losses = update_policy(model, optimizer, steps, device, normalizer)
                for tick, step in enumerate(steps):
                    logger.log_tick(episode, tick, point, step, device)
                summary = summarize_episode(steps, episode, point, device, losses)
                logger.log_episode(summary)
                progress.tick(episode, summary)
                if episode % STATUS_EVERY == 0 or not progress.is_live:
                    print(f"[relative_v10 @ {episode}] reward={reward:.1f} destroyed={destroyed} contained={contained} "
                          f"policy={losses[0]:.3f} aux={losses[3]:.3f} entropy={losses[4]:.3f}", flush=True)
                if episode % EVAL_EVERY == 0:
                    eval_rows = []
                    for name, eval_point in [("anchor", TRAINING_IGNITION_POINT), *VALIDATION_IGNITION_POINTS.items()]:
                        result = evaluate(model, env, eval_point, device)
                        row = {
                            "checkpoint_episode": episode, "scenario": name,
                            "ignition_row": eval_point[0], "ignition_col": eval_point[1],
                            "evaluation_seed": BASE_SEED, "avg_reward": result[0],
                            "avg_buildings_destroyed": result[1], "containment_rate": result[2],
                        }
                        logger.log_eval(row)
                        eval_rows.append(row)
                        print(f"  eval {name}: reward={result[0]:.1f} destroyed={result[1]:.1f} containment={result[2]:.0%}", flush=True)
                    progress.refresh_evals(eval_rows)
                if episode % CHECKPOINT_EVERY == 0:
                    checkpoint = save_checkpoint(model, episode)
                    logger.log_checkpoint(episode, checkpoint)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
