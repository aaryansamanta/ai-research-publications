"""
Runs the core Phase 2/3 comparisons (misconception proportion + concept score
gain, standard vs. enriched) per topic.

Usage:
    python analyze_pilot.py path/to/data.csv

Defaults to the synthetic placeholder dataset if no path is given.
"""

import sys
import csv
from collections import defaultdict

try:
    from scipy import stats
    import numpy as np
except ImportError:
    sys.exit(
        "Missing dependencies. Install with:\n"
        "  pip install scipy numpy --break-system-packages"
    )

DEFAULT_PATH = "../data/synthetic/synthetic_pilot_data.csv"


def load_data(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["misconception_pre"] = int(r["misconception_pre"])
            r["misconception_post"] = int(r["misconception_post"])
            r["pretest_score"] = float(r["pretest_score"])
            r["posttest_score"] = float(r["posttest_score"])
            r["confidence_rating"] = int(r["confidence_rating"])
            rows.append(r)
    return rows


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def analyze_topic(rows, topic):
    topic_rows = [r for r in rows if r["topic"] == topic]
    standard = [r for r in topic_rows if r["group"] == "standard"]
    enriched = [r for r in topic_rows if r["group"] == "enriched"]

    print(f"\n=== Topic: {topic} ===")
    print(f"n(standard)={len(standard)}  n(enriched)={len(enriched)}")

    # --- Misconception proportion: chi-square ---
    std_misconception = sum(r["misconception_post"] for r in standard)
    enr_misconception = sum(r["misconception_post"] for r in enriched)
    std_ok = len(standard) - std_misconception
    enr_ok = len(enriched) - enr_misconception

    table = [[std_misconception, std_ok], [enr_misconception, enr_ok]]
    chi2, p_chi2, dof, expected = stats.chi2_contingency(table)

    std_rate = std_misconception / len(standard) if standard else float("nan")
    enr_rate = enr_misconception / len(enriched) if enriched else float("nan")

    print(f"Post-test misconception rate — standard: {std_rate:.2%}, enriched: {enr_rate:.2%}")
    print(f"Chi-square: chi2={chi2:.3f}, p={p_chi2:.4f}")

    # --- Concept score gain: t-test + Cohen's d ---
    std_gain = [r["posttest_score"] - r["pretest_score"] for r in standard]
    enr_gain = [r["posttest_score"] - r["pretest_score"] for r in enriched]

    t_stat, p_ttest = stats.ttest_ind(enr_gain, std_gain, equal_var=False)
    d = cohens_d(enr_gain, std_gain)

    print(f"Mean gain — standard: {np.mean(std_gain):.3f}, enriched: {np.mean(enr_gain):.3f}")
    print(f"t-test: t={t_stat:.3f}, p={p_ttest:.4f}, Cohen's d={d:.3f}")

    # Non-parametric fallback
    u_stat, p_mw = stats.mannwhitneyu(enr_gain, std_gain, alternative="two-sided")
    print(f"Mann-Whitney U (fallback): U={u_stat:.1f}, p={p_mw:.4f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    rows = load_data(path)
    topics = sorted(set(r["topic"] for r in rows))
    print(f"Loaded {len(rows)} rows from {path}")
    for topic in topics:
        analyze_topic(rows, topic)


if __name__ == "__main__":
    main()
