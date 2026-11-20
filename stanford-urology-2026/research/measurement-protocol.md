# Measurement protocol and proposed reproducibility workflow

**Status:** methodological proposal, not a claim of completed analysis.

## Acquisition and provenance
1. Obtain documented permission for each recording and verify de-identification, acquisition geometry, orientation, probe settings and timestamps.
2. Identify a stable reference landmark (e.g., pubic symphysis when visible) and document how the reference is tracked.
3. Record which anatomical landmarks are visible, definitions, frame-level confidence, and exclusions.
4. Determine image spatial calibration in mm/pixel and actual frame timestamps; do not assume nominal FPS.
5. Document maneuver onset and phase: voluntary contraction, cough, Valsalva.

## Analysis
- Annotate landmark centers frame-by-frame with blinded duplicate review on a subset.
- Convert image coordinates into a reference-aligned coordinate system; define cranial/caudal sign.
- Compute Δr(t), v(t), a(t) and summarize peak excursions and timing.
- Apply and report an appropriate smoothing/filtering strategy before differentiation; compare sensitivity to its parameters.
- Quantify inter-rater and intra-rater variability, missing landmarks, pixel uncertainty, and calibration uncertainty.
- Analyze within-subject differences with a prespecified statistical plan; do not infer clinical effects from visual overlays alone.

## Minimum reporting checklist
| Field | Required provenance |
|---|---|
| Cohort | Inclusion/exclusion, sample size and data permissions |
| Acquisition | Probe frequency, orientation, frame rate and calibration |
| Annotation | Landmark definitions, annotator, software and QC |
| Maneuvers | Timing, repetitions and consistency |
| Kinematics | Coordinate frame, smoothing and differentiation |
| Uncertainty | Confidence intervals, inter-rater error and missingness |
| Clinical interpretation | Comparison groups, outcomes, confounders and limitations |

The original deck documents physician-led offline analysis and preliminary findings; it does not supply enough raw numerical data to execute this full protocol.
