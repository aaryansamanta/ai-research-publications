"""
Generates FAKE/PLACEHOLDER pilot data shaped like what Phase 2/3 data collection
will produce, so the analysis pipeline in /analysis can be built and tested
before real classroom data exists.

This is NOT real study data. Replace with actual collected data before
drawing any conclusions.
"""

import csv
import random

random.seed(42)

TOPICS = ["cell_structure", "central_dogma", "life_cycle"]
COHORTS = ["high_school", "college"]
GROUPS = ["standard", "enriched"]

N_PER_GROUP = 60  # placeholder sample size; adjust to actual recruitment


def simulate_student(student_id, group, topic, cohort):
    # Baseline misconception rate ~60% regardless of group (pre-test, before intervention)
    misconception_pre = 1 if random.random() < 0.60 else 0

    # Enriched diagrams assumed (for placeholder purposes) to reduce misconception
    # likelihood post-intervention; standard diagrams show little change.
    if group == "enriched":
        post_prob = 0.30
    else:
        post_prob = 0.55
    misconception_post = 1 if random.random() < post_prob else 0

    pretest_score = round(random.uniform(0.3, 0.7), 2)
    gain = random.uniform(0.05, 0.25) if group == "enriched" else random.uniform(-0.05, 0.15)
    posttest_score = round(min(1.0, pretest_score + gain), 2)

    confidence = random.randint(2, 5)

    return {
        "student_id": student_id,
        "cohort": cohort,
        "group": group,
        "topic": topic,
        "misconception_pre": misconception_pre,
        "misconception_post": misconception_post,
        "pretest_score": pretest_score,
        "posttest_score": posttest_score,
        "confidence_rating": confidence,
    }


def main():
    rows = []
    sid = 1
    for topic in TOPICS:
        for group in GROUPS:
            for _ in range(N_PER_GROUP):
                cohort = random.choices(COHORTS, weights=[0.8, 0.2])[0]
                rows.append(simulate_student(sid, group, topic, cohort))
                sid += 1

    fieldnames = list(rows[0].keys())
    with open("synthetic_pilot_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows to synthetic_pilot_data.csv")


if __name__ == "__main__":
    main()
