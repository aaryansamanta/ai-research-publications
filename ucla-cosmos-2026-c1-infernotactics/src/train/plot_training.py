"""Create a compact training dashboard from a RunLogger directory."""

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from data_pipeline.config import PROJECT_ROOT


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def values(rows, key):
    parsed = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        if str(value).lower() in ("true", "false"):
            value = 1.0 if str(value).lower() == "true" else 0.0
        parsed.append(float(value))
    return np.array(parsed, dtype=float)


def moving_average(x, window=20):
    if len(x) < 2:
        return x
    window = min(window, len(x))
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    run_dir = os.path.join(PROJECT_ROOT, "logs", "runs", args.run_tag)
    train_path = os.path.join(run_dir, "train_episode.csv")
    eval_path = os.path.join(run_dir, "eval.csv")
    if not os.path.exists(train_path):
        raise SystemExit(f"Training log not found: {train_path}")
    train = read_csv(train_path)
    evaluations = read_csv(eval_path) if os.path.exists(eval_path) else []
    episodes = np.arange(1, len(train) + 1)
    out_dir = os.path.join(PROJECT_ROOT, "reports", args.run_tag)
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(5, 3, figsize=(18, 22), constrained_layout=True)
    plots = [
        ("total_reward", "Total Reward", True),
        ("buildings_destroyed", "Buildings Destroyed", False),
        ("contained", "Containment (episode)", False),
        ("n_ticks", "Episode Length", False),
        ("policy_loss", "Policy Loss", False),
        ("value_loss", "Value Loss", True),
        ("classification_loss", "Classification Loss", False),
        ("aux_target_loss", "Aux Target Loss", False),
        ("entropy", "Entropy", False),
        ("actions_per_tick", "Actions per Tick", False),
        ("noop_ticks", "No-op Ticks", False),
        ("mean_response_ticks", "Mean Response Ticks", False),
        # v11 reward components
        ("rc_fire_penalty", "Fire Penalty / Episode", False),
        ("rc_dispatch_cost", "Dispatch Cost / Episode", False),
        ("rc_trench_bonus", "Trench Bonus / Episode", False),
        ("peak_active_fire", "Peak Active Fire Cells", False),
    ]
    for ax, (key, title, log_y) in zip(axes.flat, plots):
        data = values(train, key)
        x = episodes[:len(data)]
        if len(data):
            ax.plot(x, data, alpha=0.35, linewidth=1, label="episode")
            avg = moving_average(data)
            ax.plot(x[len(x) - len(avg):], avg, linewidth=2, label="moving average")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.25)
        if log_y and len(data) and np.all(data > 0):
            ax.set_yscale("log")
        ax.legend(loc="best", fontsize=8)

    if evaluations:
        eval_ax = fig.add_axes([0.10, 0.01, 0.80, 0.04])
        for scenario in sorted({row["scenario"] for row in evaluations}):
            rows = [row for row in evaluations if row["scenario"] == scenario]
            x = values(rows, "checkpoint_episode")
            y = values(rows, "avg_reward")
            eval_ax.plot(x, y, marker="o", label=scenario)
        eval_ax.set_title("Evaluation Reward by Checkpoint")
        eval_ax.set_xlabel("Checkpoint Episode")
        eval_ax.grid(alpha=0.25)
        eval_ax.legend(fontsize=8)

    png_path = os.path.join(out_dir, "training_dashboard.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    summary = {
        "run_tag": args.run_tag,
        "episodes_logged": len(train),
        "last_reward": float(values(train, "total_reward")[-1]) if train else None,
        "last_buildings_destroyed": float(values(train, "buildings_destroyed")[-1]) if train else None,
        "dashboard": png_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(png_path)


if __name__ == "__main__":
    main()
