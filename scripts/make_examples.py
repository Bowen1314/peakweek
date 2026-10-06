"""Run forecast() for the example places and save examples/forecast_<slug>.json.

  python scripts/make_examples.py                  # all three, today's date (America/New_York)
  python scripts/make_examples.py --warm-cache     # only fetch weather + elevation (no TabPFN)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from peakweek import dataset, weather  # noqa: E402

PLACES = {
    "new_brunswick": ("New Brunswick, NJ", 40.4862, -74.4518),
    "burlington": ("Burlington, VT", 44.4759, -73.2121),
    "pittsburgh": ("Pittsburgh, PA", 40.4406, -79.9959),
}


def maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if platform.system() == "Darwin" else r / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=None)
    ap.add_argument("--warm-cache", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    today = dt.date.fromisoformat(args.today) if args.today else weather.local_today()
    places = {k: v for k, v in PLACES.items() if not args.only or k in args.only}
    if args.warm_cache:
        clim = dataset.load_climatology()
        for slug, (name, lat, lon) in places.items():
            wlat, wlon, cid, known = dataset.weather_point(lat, lon, clim)
            _, meta = weather.app_daily(wlat, wlon, today)
            elev = weather.elevations([(lat, lon)])[0]
            print(f"{slug}: cell {cid} (in training climatology: {known}) weather point {wlat},{wlon} "
                  f"elevation {elev} weather {meta}")
        print(json.dumps(weather.usage_summary()))
        return
    from peakweek.forecast import forecast
    timings = {}
    for slug, (name, lat, lon) in places.items():
        t = time.time()
        res = forecast(lat, lon, name, today)
        wall = time.time() - t
        path = ROOT / "examples" / f"forecast_{slug}.json"
        path.write_text(json.dumps(res, indent=1))
        timings[slug] = {"wall_seconds": round(wall, 1), "model_seconds": res["model"]["seconds"],
                         "maxrss_mb_so_far": round(maxrss_mb(), 1)}
        print(slug, json.dumps(timings[slug]), flush=True)
    (ROOT / "examples" / "forecast_timings.json").write_text(json.dumps(
        {"today": today.isoformat(), "host": platform.node(), "timings": timings}, indent=1))


if __name__ == "__main__":
    main()
