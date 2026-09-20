"""Crash-safe CSV/JSONL/TensorBoard logging for training runs."""

import csv
import json
import os
import subprocess
from datetime import datetime, timezone


EPISODE_FIELDS = [
    "episode", "ignition_row", "ignition_col", "seed", "device", "n_ticks",
    "total_reward", "mean_reward_per_tick", "buildings_destroyed", "contained", "timeout",
    "first_building_loss_tick", "peak_active_fire", "final_active_fire", "peak_blaze",
    "policy_loss", "value_loss", "classification_loss", "aux_target_loss", "entropy",
    "resource_entropy", "target_entropy", "grad_norm", "actions_per_tick", "max_actions_tick",
    "noop_ticks", "dispatch_attempts", "successful_dispatches", "wasted_dispatches",
    "unavailable_resource_attempts", "unreachable_zone_attempts", "mean_response_ticks",
    "mean_traffic_delay_s", "traffic_mean_load", "traffic_max_load",
    "water_team_dispatches", "trench_crew_dispatches", "rescue_vehicle_dispatches", "helicopter_dispatches",
    "water_team_effect_success", "trench_crew_effect_success", "rescue_vehicle_effect_success", "helicopter_effect_success",
    "rc_fire_extinguished", "rc_fire_penalty", "rc_dispatch_cost",
    "rc_trench_bonus", "rc_buildings_destroyed", "rc_travel_delay", "rc_wasted",
    "trench_held_total",
]

EVAL_FIELDS = [
    "checkpoint_episode", "scenario", "ignition_row", "ignition_col", "evaluation_seed",
    "avg_reward", "avg_buildings_destroyed", "containment_rate", "avg_ticks",
    "dispatch_attempts", "successful_dispatches", "wasted_dispatches", "mean_actions_per_tick",
    "water_team_dispatches", "trench_crew_dispatches", "rescue_vehicle_dispatches", "helicopter_dispatches",
]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


class RunLogger:
    def __init__(self, project_root, run_tag, config, trace_every=0):
        self.run_tag = run_tag
        self.trace_every = int(trace_every)
        self.run_dir = os.path.join(project_root, "logs", "runs", run_tag)
        os.makedirs(self.run_dir, exist_ok=True)
        manifest = dict(config)
        manifest.update({"run_tag": run_tag, "started_at": datetime.now(timezone.utc).isoformat()})
        manifest["git_commit"] = self._git_commit(project_root)
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        self.episode_file = open(os.path.join(self.run_dir, "train_episode.csv"), "w", newline="", encoding="utf-8")
        self.eval_file = open(os.path.join(self.run_dir, "eval.csv"), "w", newline="", encoding="utf-8")
        self.checkpoint_file = open(os.path.join(self.run_dir, "checkpoints.csv"), "w", newline="", encoding="utf-8")
        self.episode_writer = csv.DictWriter(self.episode_file, fieldnames=EPISODE_FIELDS, extrasaction="ignore")
        self.eval_writer = csv.DictWriter(self.eval_file, fieldnames=EVAL_FIELDS, extrasaction="ignore")
        self.checkpoint_writer = csv.DictWriter(self.checkpoint_file, fieldnames=["episode", "path"], extrasaction="ignore")
        self.episode_writer.writeheader(); self.eval_writer.writeheader(); self.checkpoint_writer.writeheader()
        self.trace_file = open(os.path.join(self.run_dir, "train_tick.jsonl"), "w", encoding="utf-8") if self.trace_every else None
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(project_root, "logs", "tensorboard", run_tag))
        except Exception:
            pass

    @staticmethod
    def _git_commit(project_root):
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
        except Exception:
            return "unknown"

    def log_tick(self, episode, tick, ignition, step, device):
        if self.trace_file is None or (episode % self.trace_every != 0):
            return
        info = step.get("info", {})
        dispatches = info.get("dispatch", [])
        row = {
            "run_tag": self.run_tag, "episode": episode, "tick": tick,
            "ignition": list(ignition), "device": str(device),
            "state_counts": info.get("state_counts", {}),
            "buildings_destroyed": info.get("buildings_destroyed", 0),
            "reward": step.get("reward", 0.0),
            "actions": step.get("actions", []), "dispatches": dispatches,
            "traffic_mean_load": step["scalars"][8] if len(step.get("scalars", [])) > 8 else 0.0,
            "traffic_max_load": step["scalars"][9] if len(step.get("scalars", [])) > 9 else 0.0,
        }
        self.trace_file.write(json.dumps(row, default=str) + "\n")

    def log_episode(self, row):
        self.episode_writer.writerow({k: row.get(k, "") for k in EPISODE_FIELDS})
        self.episode_file.flush()
        if self.writer:
            episode = row["episode"]
            for key in ("total_reward", "buildings_destroyed", "mean_reward_per_tick", "contained", "n_ticks", "actions_per_tick", "noop_ticks", "mean_response_ticks", "mean_traffic_delay_s"):
                value = row.get(key)
                if value != "" and value is not None:
                    self.writer.add_scalar(f"train/{key}", float(value), episode)
            for key in ("policy_loss", "value_loss", "classification_loss", "aux_target_loss", "entropy", "resource_entropy", "target_entropy", "grad_norm"):
                if key in row and row[key] != "" and row[key] is not None:
                    self.writer.add_scalar(f"loss/{key}" if "loss" in key else f"policy/{key}", float(row[key]), episode)
            self.writer.flush()

    def log_eval(self, row):
        self.eval_writer.writerow({k: row.get(k, "") for k in EVAL_FIELDS})
        self.eval_file.flush()
        if self.writer and row.get("checkpoint_episode") is not None:
            step = int(row["checkpoint_episode"])
            for key in ("avg_reward", "avg_buildings_destroyed", "containment_rate", "avg_ticks"):
                if key in row:
                    self.writer.add_scalar(f"eval/{row['scenario']}/{key}", float(row[key]), step)
            self.writer.flush()

    def log_checkpoint(self, episode, path):
        self.checkpoint_writer.writerow({"episode": episode, "path": path})
        self.checkpoint_file.flush()

    def close(self):
        for f in (self.episode_file, self.eval_file, self.checkpoint_file, self.trace_file):
            if f:
                f.close()
        if self.writer:
            self.writer.close()


def summarize_episode(steps, episode, ignition, device, losses):
    infos = [s.get("info", {}) for s in steps]
    dispatches = [d for info in infos for d in info.get("dispatch", [])]
    actions = [a for s in steps for a in s.get("actions", [])]
    counts = [i.get("state_counts", {}) for i in infos]
    successful = [d for d in dispatches if d.get("status") == "dispatched"]
    wasted = [d for d in dispatches if d.get("status") != "dispatched"]
    effect_success = {rtype: 0 for rtype in ("water_team", "trench_crew", "rescue_vehicle", "helicopter")}
    for info in infos:
        for event in info.get("resource_events", []):
            if event.get("success"):
                effect_success[event["resource_type"]] += 1
    first_loss = next((i for i, info in enumerate(infos) if info.get("buildings_destroyed", 0) > 0), "")
    active = [c.get("Threat", 0) + c.get("Blaze", 0) for c in counts]
    blaze = [c.get("Blaze", 0) for c in counts]
    # Aggregate v11 reward components across the episode
    rc_keys = ("fire_extinguished", "fire_penalty", "dispatch_cost",
               "trench_bonus", "buildings_destroyed", "travel_delay", "wasted")
    rc_sums = {k: 0.0 for k in rc_keys}
    for info in infos:
        rc = info.get("reward_components", {})
        for k in rc_keys:
            if k in rc:
                rc_sums[k] += rc[k]
    trench_held_total = sum(info.get("trench_held", 0) for info in infos)
    return {
        "episode": episode, "ignition_row": ignition[0], "ignition_col": ignition[1],
        "device": str(device), "n_ticks": len(steps),
        "total_reward": sum(s.get("reward", 0.0) for s in steps),
        "mean_reward_per_tick": _mean([s.get("reward", 0.0) for s in steps]),
        "buildings_destroyed": sum(i.get("buildings_destroyed", 0) for i in infos),
        "contained": bool(infos[-1].get("contained", False)) if infos else False,
        "timeout": bool(infos[-1].get("timeout", False)) if infos else False,
        "first_building_loss_tick": first_loss, "peak_active_fire": max(active, default=0),
        "final_active_fire": active[-1] if active else 0, "peak_blaze": max(blaze, default=0),
        "policy_loss": losses[0], "value_loss": losses[1], "classification_loss": losses[2],
        "aux_target_loss": losses[3], "entropy": losses[4], "resource_entropy": losses[5],
        "target_entropy": losses[6], "grad_norm": losses[7] if len(losses) > 7 else "",
        "actions_per_tick": len(actions) / max(1, len(steps)), "max_actions_tick": max((len(s.get("actions", [])) for s in steps), default=0),
        "noop_ticks": sum(not s.get("actions") for s in steps),
        "dispatch_attempts": len(dispatches), "successful_dispatches": len(successful), "wasted_dispatches": len(wasted),
        "unavailable_resource_attempts": sum(d.get("status") == "no_unit_available" for d in dispatches),
        "unreachable_zone_attempts": sum(d.get("status") == "zone_unreachable" for d in dispatches),
        "mean_response_ticks": _mean([d.get("eta_ticks", 0) for d in successful]),
        "mean_traffic_delay_s": _mean([d.get("traffic_delay_s", 0) for d in successful]),
        "traffic_mean_load": _mean([s.get("scalars", [0] * 9)[8] for s in steps]),
        "traffic_max_load": max((s.get("scalars", [0] * 10)[9] for s in steps), default=0),
        "rc_fire_extinguished": rc_sums["fire_extinguished"],
        "rc_fire_penalty": rc_sums["fire_penalty"],
        "rc_dispatch_cost": rc_sums["dispatch_cost"],
        "rc_trench_bonus": rc_sums["trench_bonus"],
        "rc_buildings_destroyed": rc_sums["buildings_destroyed"],
        "rc_travel_delay": rc_sums["travel_delay"],
        "rc_wasted": rc_sums["wasted"],
        "trench_held_total": trench_held_total,
    } | {f"{rtype}_dispatches": sum(d.get("resource_type") == rtype for d in successful) for rtype in effect_success} \
      | {f"{rtype}_effect_success": effect_success[rtype] for rtype in effect_success}
