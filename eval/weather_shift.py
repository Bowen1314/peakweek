"""Training uses ERA5 (Open-Meteo archive API); the app uses the forecast API's past_days (model analyses).
This compares the two for Sep 2026 at a few training weather points and reports daily differences.
Cost: ~2.2 weighted calls (archive, 30 days) per point; forecast responses are shared with the app cache.

  python eval/weather_shift.py --today 2026-10-06
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from peakweek import dataset, weather  # noqa: E402

POINTS = {"New Brunswick NJ": (40.4862, -74.4518), "Burlington VT": (44.4759, -73.2121),
          "Pittsburgh PA": (40.4406, -79.9959), "Boston MA": (42.3601, -71.0589), "Albany NY": (42.6526, -73.7562)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", default=None)
    args = ap.parse_args()
    today = dt.date.fromisoformat(args.today) if args.today else weather.local_today()
    start, end = dt.date(today.year, 9, 1), dt.date(today.year, 9, 30)
    clim = dataset.load_climatology()
    rows = []
    for name, (lat, lon) in POINTS.items():
        wlat, wlon, cid, known = dataset.weather_point(lat, lon, clim)
        a = weather.archive_daily(wlat, wlon, start, end).set_index("date")
        f = weather.forecast_daily(wlat, wlon, 92, 16, run_date=today).set_index("date")
        j = a.join(f, lsuffix="_era5", rsuffix="_fc", how="inner").dropna()
        for v in ("tmean", "tmin", "prcp"):
            d = j[f"{v}_fc"] - j[f"{v}_era5"]
            rows.append({"place": name, "cell": cid, "var": v, "days": len(d), "mean_diff_fc_minus_era5": round(d.mean(), 2),
                         "mean_abs_diff": round(d.abs().mean(), 2)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    summ = df.groupby("var")[["mean_diff_fc_minus_era5", "mean_abs_diff"]].mean().round(2)
    print(summ.to_string())
    out = ROOT / "eval" / "live" / f"weather_shift_{today}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"window": [str(start), str(end)], "rows": rows,
                               "mean_by_var": summ.to_dict(orient="index")}, indent=1))


if __name__ == "__main__":
    main()
