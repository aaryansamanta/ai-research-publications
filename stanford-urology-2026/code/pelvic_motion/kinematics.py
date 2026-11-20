"""Kinematics from manually supplied landmark tracks.

No clinical measurements are included. Pixel-to-mm calibration and timebase
are required before physical interpretation. Acceleration is noise-sensitive.
"""
from __future__ import annotations
import math
from typing import Sequence

def validate_track(times: Sequence[float], x: Sequence[float], y: Sequence[float]) -> None:
    if not (len(times) == len(x) == len(y)) or len(times) < 3:
        raise ValueError("At least three synchronized observations are required")
    if not all(math.isfinite(float(v)) for series in (times, x, y) for v in series):
        raise ValueError("Track contains non-finite values")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Time values must increase strictly")

def _derivative(values: Sequence[float], times: Sequence[float]) -> list[float]:
    n = len(values)
    result = [(values[1] - values[0]) / (times[1] - times[0])]
    for i in range(1, n - 1):
        # Three-point unequal-time quadratic derivative (reduces to central difference for uniform sampling).
        h0, h1 = times[i] - times[i - 1], times[i + 1] - times[i]
        result.append((-h1 / (h0 * (h0 + h1))) * values[i - 1]
                      + ((h1 - h0) / (h0 * h1)) * values[i]
                      + (h0 / (h1 * (h0 + h1))) * values[i + 1])
    result.append((values[-1] - values[-2]) / (times[-1] - times[-2]))
    return result

def calculate_kinematics(times: Sequence[float], x: Sequence[float], y: Sequence[float],
                         mm_per_pixel: float | None = None) -> list[dict]:
    """Return position, baseline-relative displacement, velocity, acceleration.

    x/y are image pixel coordinates (positive y downward). With calibration,
    physical coordinates are in millimeters; otherwise in pixels. No filtering
    or anatomical registration is performed by this function.
    """
    validate_track(times, x, y)
    if mm_per_pixel is not None and (not math.isfinite(mm_per_pixel) or mm_per_pixel <= 0):
        raise ValueError("mm_per_pixel must be a positive finite number")
    scale = mm_per_pixel if mm_per_pixel is not None else 1.0
    xx, yy = [float(v)*scale for v in x], [float(v)*scale for v in y]
    vx, vy = _derivative(xx, times), _derivative(yy, times)
    ax, ay = _derivative(vx, times), _derivative(vy, times)
    return [{"time_s": float(t), "x": xx[i], "y": yy[i],
             "dx": xx[i]-xx[0], "dy": yy[i]-yy[0],
             "displacement_magnitude": math.hypot(xx[i]-xx[0], yy[i]-yy[0]),
             "vx": vx[i], "vy": vy[i], "speed": math.hypot(vx[i],vy[i]),
             "ax": ax[i], "ay": ay[i], "acceleration_magnitude": math.hypot(ax[i],ay[i])}
            for i,t in enumerate(times)]

def summarize_track(rows: Sequence[dict]) -> dict:
    if not rows: raise ValueError("No rows")
    return {"frames": len(rows), "duration_s": rows[-1]["time_s"]-rows[0]["time_s"],
            "peak_displacement": max(r["displacement_magnitude"] for r in rows),
            "peak_speed": max(r["speed"] for r in rows)}
