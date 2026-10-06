"""Build data/observations.csv, data/features.csv and data/climatology.csv.

Usage:
  python scripts/build_dataset.py --plan-only     # iNat fetch (cached) + Open-Meteo budget estimate
  python scripts/build_dataset.py                 # full build (Open-Meteo fetches are cached in .cache/)
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

import pandas as pd  # noqa: E402

from peakweek import config, dataset, inat, weather  # noqa: E402
from peakweek.species import BY_TAXON, TAXON_IDS  # noqa: E402

D1, D2 = "2018-01-01", "2025-12-31"
MONTHS = "9,10,11"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--max-weight", type=float, default=7000.0,
                    help="refuse to start weather fetching if the plan exceeds this many weighted calls")
    args = ap.parse_args()
    t0 = time.time()

    records, drops = inat.fetch_labeled(TAXON_IDS, D1, D2, MONTHS)
    obs_all = dataset.observations_frame(records)
    obs, n_unlicensed = dataset.license_filter(obs_all)
    print(f"labeled observations: {len(obs_all)}; dropped for null license_code: {n_unlicensed}; kept {len(obs)}")

    cells = dataset.assign_cells(obs, config.GRID_DEG)
    plan = dataset.weather_plan(cells)
    print(json.dumps({k: v for k, v in plan.items() if k != "cells"}, indent=1))
    if args.plan_only:
        return
    if plan["weight_full"] > args.max_weight:
        raise SystemExit(f"weather plan {plan['weight_full']} weighted calls exceeds --max-weight {args.max_weight}")

    # Elevation at the observation point (3-decimal coordinates, SRTM 90 m via OpenTopoData; see
    # weather.elevations_srtm for why not Open-Meteo).
    coords = list(zip(cells["lat"], cells["lon"]))
    elev = weather.elevations_srtm(coords, decimals=3)
    elevation = dict(zip(cells["obs_id"], elev))
    print(f"elevation: {len(set(coords))} unique points, {sum(e is None for e in elev)} missing")

    table, clim = dataset.build_feature_table(cells, elevation)
    config.DATA_DIR.mkdir(exist_ok=True)
    obs.to_csv(config.DATA_DIR / "observations.csv", index=False)
    table.to_csv(config.DATA_DIR / "features.csv", index=False, float_format="%.4f")
    clim.to_csv(config.DATA_DIR / "climatology.csv", index=False, float_format="%.4f")

    # Summary for data/README.md and the report.
    counts = (table.assign(common=table["taxon_id"].map(lambda t: BY_TAXON[t]["common"]))
              .pivot_table(index=["common", "year"], columns="label", values="obs_id", aggfunc="count", fill_value=0))
    by_species = table.assign(common=table["taxon_id"].map(lambda t: BY_TAXON[t]["common"])).pivot_table(
        index="common", columns="label", values="obs_id", aggfunc="count", fill_value=0)
    by_year = table.pivot_table(index="year", columns="label", values="obs_id", aggfunc="count", fill_value=0)
    drop_tot: dict = {}
    for d in drops.values():
        for k, v in d.items():
            drop_tot[k] = drop_tot.get(k, 0) + v
    summary = {
        "pulled_on": dt.date.today().isoformat(),
        "inat_query": {"term_id": 36, "d1": D1, "d2": D2, "month": MONTHS, "bbox": config.REGION,
                       "taxa": TAXON_IDS},
        "inat_outcomes_total": drop_tot,
        "inat_outcomes_by_taxon": {str(k): v for k, v in drops.items()},
        "labeled": len(obs_all), "dropped_null_license": n_unlicensed, "kept": len(obs),
        "feature_rows": len(table),
        "grid_deg": config.GRID_DEG, "n_cells": plan["n_cells"], "n_cell_years": plan["n_cell_years"],
        "openmeteo": weather.usage_summary(),
        "by_species": by_species.to_dict(orient="index"),
        "by_year": {str(k): v for k, v in by_year.to_dict(orient="index").items()},
        "seconds": round(time.time() - t0, 1),
    }
    (config.DATA_DIR / "build_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(by_species.to_string())
    print(by_year.to_string())
    print(counts.to_string())
    print(json.dumps(summary["openmeteo"], indent=1))


if __name__ == "__main__":
    main()
