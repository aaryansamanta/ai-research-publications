# Analysis Plan

Matches the statistical approach described in the manuscript's Goals and Evaluation section.

## 1. Misconception Proportion (categorical outcome)

- **Test:** Chi-square test of independence (or Fisher's exact test if any expected cell count < 5)
- **Comparison:** `misconception_post` rate, standard group vs. enriched group, per topic
- **Also report:** pre → post change within each group (McNemar's test, since it's the same students measured twice)

## 2. Concept Score Gains (continuous outcome)

- **Compute:** gain score = `posttest_score - pretest_score` per student
- **Test:** Independent-samples t-test (standard vs. enriched) if gain scores are approximately normal; Mann-Whitney U as the non-parametric fallback
- **Effect size:** Cohen's d, target ≥ 0.4 per proposal
- Consider ANCOVA (post-test score as outcome, pre-test score as covariate, group as factor) as a more robust alternative to raw gain scores if pre-test scores differ meaningfully between groups

## 3. Qualitative Coding

- Drawing-to-explain scores (see `instruments/drawing_task.md`) coded blind to group
- Inter-rater reliability (Cohen's kappa) on a ≥20% double-coded subsample
- Compare rubric sub-scores (structural accuracy, dynamics/variation, causal/sequential accuracy) between groups using the same tests as above

## 4. Reporting

For each topic and outcome, report: n per group, descriptive stats, test statistic, p-value, effect size, and a plain-language interpretation. Correct for multiple comparisons (e.g., Holm-Bonferroni) if testing several topics/outcomes as a family.

## 5. Pipeline

`analyze_pilot.py` runs steps 1–2 on whatever CSV is passed to it (synthetic placeholder now, real data later) and prints a summary table per topic.
