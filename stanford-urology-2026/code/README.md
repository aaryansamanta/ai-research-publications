# kinematic methods



## Run (Python 3.10+, standard library only)

```bash
cd code
python -m unittest discover -s tests -v
python -m pelvic_motion.cli examples/synthetic_landmark_track.csv examples/synthetic_output.csv
# Only if calibration is independently known:
python -m pelvic_motion.cli examples/synthetic_landmark_track.csv examples/synthetic_output_mm.csv --mm-per-pixel 0.15
```

Input columns: `time_s,x_px,y_px`. The example is **synthetic**. Output includes baseline-relative `dx,dy`, displacement magnitude, `vx,vy`, speed, `ax,ay`, acceleration magnitude. Pixel units remain pixels without calibration. Image y increases downward; physical cranial/caudal interpretation requires verified image orientation.

## Equations

For landmark position **r(t) = [x(t), y(t)]**, baseline **r₀**, displacement **Δr(t)=r(t)−r₀**, velocity **v(t)=dr/dt**, and acceleration **a(t)=d²r/dt²**. Unequally spaced interior samples use a three-point quadratic derivative. Boundary samples use one-sided differences. Derivatives amplify tracking noise; filtering, error bars, calibration, reference registration and quality checks are needed before research inference.

