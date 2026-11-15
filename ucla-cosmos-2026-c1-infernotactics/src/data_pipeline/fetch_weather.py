"""
Pull real Jan 7-8, 2025 hourly weather observations covering the Palisades
Fire's onset, and save a clean wind/humidity time series for use by
inferno_env.py (replaces the fixed synthetic Santa-Ana-ramp placeholder).

Source: NOAA ASOS via the Iowa Environmental Mesonet's public historical-data
API (no auth needed) -- https://mesonet.agron.iastate.edu/request/download.phtml
Station: KSMO, Santa Monica Municipal Airport -- chosen over Van Nuys (KVNY)
because it sits only ~7km south of Pacific Palisades (inside/adjacent to our
study bbox), vs. KVNY's ~20km away on the far side of the Santa Monica
Mountains, so it's much more representative of conditions at the fire.

The pulled window (Jan 7 00:00 UTC - Jan 9 00:00 UTC) covers the fire's
actual ignition (~10:30am PST / 18:30 UTC Jan 7, see FIRE_START_UTC in
inferno_env.py) with a comfortable buffer on both sides.
"""

import csv
import os
import sys
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import DATA_DIR, WEATHER_CSV_PATH, WEATHER_STATION  # noqa: E402

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
KNOTS_TO_MPH = 1.15078

# Pull window: comfortably brackets the fire's real ignition time with buffer
# on both sides, in case an episode's real-weather anchor is nudged around it.
PULL_START = (2025, 1, 7)
PULL_END = (2025, 1, 9)  # exclusive-ish; IEM includes the full end day


def fetch_weather():
    os.makedirs(DATA_DIR, exist_ok=True)

    params = {
        "station": WEATHER_STATION,
        "data": "relh,drct,sknt",
        "year1": PULL_START[0], "month1": PULL_START[1], "day1": PULL_START[2],
        "year2": PULL_END[0], "month2": PULL_END[1], "day2": PULL_END[2],
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        # Routine + special hourly METAR reports only -- without this filter
        # IEM also returns 5-minute-interval readings that carry wind but not
        # humidity, which would otherwise dominate and get skipped below.
        "report_type": [3, 4],
    }
    print(f"Fetching ASOS observations for station {WEATHER_STATION}, "
          f"{PULL_START} to {PULL_END} (UTC) from IEM...")
    resp = requests.get(IEM_ASOS_URL, params=params, timeout=60)
    resp.raise_for_status()

    lines = resp.text.strip().splitlines()
    reader = csv.DictReader(lines)

    rows = []
    n_skipped = 0
    for rec in reader:
        try:
            wind_speed_kt = float(rec["sknt"])
            wind_dir_deg = float(rec["drct"])
            humidity_pct = float(rec["relh"])
        except ValueError:
            n_skipped += 1  # missing ("M") reading for a field we need
            continue
        ts = datetime.strptime(rec["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        rows.append((ts, wind_speed_kt * KNOTS_TO_MPH, wind_dir_deg, humidity_pct))

    rows.sort(key=lambda r: r[0])
    print(f"Parsed {len(rows)} usable hourly observations ({n_skipped} skipped for missing data)")

    with open(WEATHER_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "wind_speed_mph", "wind_direction_deg", "humidity_pct"])
        for ts, wind_mph, wind_dir, humidity in rows:
            writer.writerow([ts.strftime("%Y-%m-%dT%H:%M:%SZ"), f"{wind_mph:.2f}", f"{wind_dir:.1f}", f"{humidity:.2f}"])

    print(f"Saved {len(rows)} rows to {WEATHER_CSV_PATH}")

    # Sanity check: the Palisades Fire is well documented as an extreme,
    # single-digit-humidity Santa Ana wind event -- confirm the pulled data
    # actually shows that, not e.g. a calm/wet stretch from a bad date range.
    min_humidity = min(r[3] for r in rows)
    max_wind = max(r[1] for r in rows)
    print(f"\nSanity check: min humidity={min_humidity:.1f}%  max wind={max_wind:.1f} mph "
          f"(expect single-digit humidity and 30+ mph wind for a real Santa Ana event)")

    return rows


if __name__ == "__main__":
    fetch_weather()
