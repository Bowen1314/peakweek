"""Live check: compare predictions with this season's iNaturalist observations (all after the training data).

Fetches Leaves-annotated observations of the 8 species in the training box made in the last --days days,
builds features exactly like the app (weather.app_daily at the cell's weather point: ERA5 archive through
yesterday + forecast API; all-years climatology; point elevation), predicts with the locked TabPFN config trained on 2018-2025, and
compares with the climatology / logistic / HGB baselines trained on the same rows.

  python eval/live_check.py                 # prepare + predict (needs TabPFN, i.e. run on the Dell)
  python eval/live_check.py --stage prepare # fetch + features only (no TabPFN)
  python eval/live_check.py --stage predict --rows eval/live/rows_2026-10-06.csv

Results: eval/live/live_<today>.json (n, metrics, per-species) and per-row predictions.
Note: features use weather through the day before each observation, i.e. observed (archive) weather, not
forecasts issued before the observation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from peakweek import baselines, config, dataset, features, inat, metrics, weather  # noqa: E402
from peakweek.forecast import LOCKED_CONFIG, PAST_DAYS, FORECAST_DAYS  # noqa: E402
from peakweek.model import PeakweekModel  # noqa: E402
from peakweek.species import BY_TAXON, TAXON_IDS  # noqa: E402

LIVE = ROOT / "eval" / "live"


def prepare(today: dt.date, days: int) -> Path:
    d1 = (today - dt.timedelta(days=days)).isoformat()
    records, drops = inat.fetch_labeled(TAXON_IDS, d1, today.isoformat(), months=None)
    obs = dataset.observations_frame(records)
    obs = obs[pd.to_datetime(obs["observed_on"]).dt.date < today]  # today's day is incomplete
    print(f"live observations {d1}..{today}: {len(obs)} labeled (license filter not applied)")
    clim = dataset.load_climatology()
    obs = obs.assign(cell_id=[weather.cell_id(a, b) for a, b in zip(obs["lat"], obs["lon"])])
    elev = weather.elevations_srtm(list(zip(obs["lat"], obs["lon"])), decimals=3)
    obs["elevation_m"] = elev
    parts = []
    unknown_cells = []
    for cid, grp in obs.groupby("cell_id"):
        lat0, lon0 = float(grp["lat"].iloc[0]), float(grp["lon"].iloc[0])
        wx_lat, wx_lon, _, known = dataset.weather_point(lat0, lon0, clim)
        if known:
            curve = dataset.climatology_curve(clim, cid)
        else:
            unknown_cells.append(cid)
            curves = dataset.cell_curves(wx_lat, wx_lon, config.TRAIN_YEARS)
            total, count = features.climatology_from_curves(list(curves.values()))
            curve = features.leave_one_out_mean(total, count, None)
        daily, _ = weather.app_daily(wx_lat, wx_lon, today, PAST_DAYS, FORECAST_DAYS)  # same as the app
        pts = grp.rename(columns={"observed_on": "date"})
        fr = features.feature_rows(pts, daily, curve).rename(columns={"date": "observed_on"})
        parts.append(fr)
    rows = pd.concat(parts, ignore_index=True)
    rows["label_int"] = rows["label"].map(config.LABEL_TO_INT)
    rows["year"] = pd.to_datetime(rows["observed_on"]).dt.year
    LIVE.mkdir(parents=True, exist_ok=True)
    path = LIVE / f"rows_{today}.csv"
    rows.to_csv(path, index=False, float_format="%.4f")
    meta = {"today": today.isoformat(), "d1": d1, "n": len(rows), "inat_outcomes": drops,
            "cells": int(rows["cell_id"].nunique()), "cells_without_training_climatology": unknown_cells,
            "openmeteo_usage_this_machine": weather.usage_summary()}
    (LIVE / f"rows_{today}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: v for k, v in meta.items() if k != "inat_outcomes"}, default=str))
    return path


def predict(rows_path: Path, with_tabpfn: bool = True) -> dict:
    rows = pd.read_csv(rows_path)
    rows["observed_on"] = pd.to_datetime(rows["observed_on"]).dt.date
    train = dataset.load_features()
    y = rows["label_int"].to_numpy()
    out = {"rows": str(rows_path.relative_to(ROOT)), "n": int(len(rows)), "n_train": int(len(train)),
           "label_counts": {k: int((rows["label"] == k).sum()) for k in config.LABELS},
           "date_range": [str(rows["observed_on"].min()), str(rows["observed_on"].max())], "models": {}}
    models = {"climatology": baselines.Climatology(), "logreg_calendar": baselines.LogRegCalendar(),
              "logreg_all": baselines.LogRegAll(), "hgb": baselines.HGB()}
    locked_path = ROOT / "eval" / "locked.json"
    if locked_path.exists():
        b = json.loads(locked_path.read_text())["baselines"]
        models = {"climatology": baselines.Climatology(),
                  "logreg_calendar": baselines.LogRegCalendar(C=b["logreg_calendar"]["C"]),
                  "logreg_all": baselines.LogRegAll(C=b["logreg_all"]["C"]),
                  "hgb": baselines.HGB(**{k: v for k, v in b["hgb"].items() if k not in ("features", "defaults")})}
    preds = rows[["obs_id", "observed_on", "taxon_id", "cell_id", "label_int"]].copy()
    for name, m in models.items():
        P = m.fit(train).predict_proba(rows)
        out["models"][name] = metrics.summary(y, P)
        preds[[f"{name}_p_green", f"{name}_p_colored", f"{name}_p_bare"]] = P
    if with_tabpfn:
        t = time.time()
        m = PeakweekModel(**LOCKED_CONFIG).fit(train)
        P = m.predict_proba(rows)
        out["models"]["tabpfn"] = dict(metrics.summary(y, P), seconds=round(time.time() - t, 1), config=m.config())
        preds[["tabpfn_p_green", "tabpfn_p_colored", "tabpfn_p_bare"]] = P
        out["tabpfn_per_species"] = {
            BY_TAXON[int(k)]["common"]: v
            for k, v in metrics.per_group_log_loss(y, P, rows["taxon_id"].to_numpy()).items()}
    stem = rows_path.stem.replace("rows_", "")
    preds.to_csv(LIVE / f"preds_{stem}.csv", index=False, float_format="%.6f")
    (LIVE / f"live_{stem}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "prepare", "predict"], default="all")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--today", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--no-tabpfn", action="store_true")
    args = ap.parse_args()
    today = dt.date.fromisoformat(args.today) if args.today else weather.local_today()
    path = Path(args.rows) if args.rows else None
    if args.stage in ("all", "prepare"):
        path = prepare(today, args.days)
    if args.stage in ("all", "predict"):
        path = path or LIVE / f"rows_{today}.csv"
        predict(path if path.is_absolute() else ROOT / path, with_tabpfn=not args.no_tabpfn)


if __name__ == "__main__":
    main()
