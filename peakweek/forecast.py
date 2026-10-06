"""forecast(): per-species daily leaf-state probabilities for a point, today .. today+14 (docs/CONTRACT.md).

Weather is fetched at the weather point of the point's grid cell (the same point the training rows of
that cell used; see config.GRID_DEG): the Open-Meteo forecast API (past_days=92, forecast_days=16) for
today onward, and the Open-Meteo archive (ERA5, the training source) for Sep 1 .. yesterday.
Elevation is looked up for the point itself. TabPFN is imported lazily (only when no classifier is
injected), so importing this module never imports tabpfn.
"""

from __future__ import annotations

import datetime as dt
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config, dataset, features, weather
from .model import TABPFN_NAME, PeakweekModel
from .species import SPECIES

N_DAYS = 15
PAST_DAYS = 92
FORECAST_DAYS = 16

# Locked after validation on 2024 (see eval/RESULTS.md and eval/locked.json, entry "tabpfn").
# Do not tune on 2025. tests/test_forecast.py checks that this matches eval/locked.json.
LOCKED_CONFIG = {
    "feature_set": "full",
    "strategy": "stratified",
    "n_context": 1000,
    "n_estimators": 4,
    "n_subsamples": 3,
    "seed": 0,
    "floor": 8,
}


@lru_cache(maxsize=1)
def _context_pool() -> pd.DataFrame:
    return dataset.load_features()


@lru_cache(maxsize=1)
def _climatology() -> pd.DataFrame:
    return dataset.load_climatology()


def climatology_curve_for(lat: float, lon: float, clim_table: pd.DataFrame | None = None) -> tuple[np.ndarray, str]:
    """All-years (2018-2025) climatological tmean_season curve for the point's weather cell.

    Uses data/climatology.csv; for a cell without training observations, fetches ERA5 season windows
    at the cell's geometric center (cached on disk) and computes the same curve."""
    clim_table = _climatology() if clim_table is None else clim_table
    wx_lat, wx_lon, cid, known = dataset.weather_point(lat, lon, clim_table)
    if known:
        return dataset.climatology_curve(clim_table, cid), "data/climatology.csv"
    curves = dataset.cell_curves(wx_lat, wx_lon, config.TRAIN_YEARS)
    total, count = features.climatology_from_curves(list(curves.values()))
    return features.leave_one_out_mean(total, count, None), "fetched on demand (ERA5 2018-2025, cell center)"


def _round_probs(p: np.ndarray) -> list[float]:
    r = np.round(np.asarray(p, dtype=float), 6)
    r[int(np.argmax(r))] += 1.0 - r.sum()
    return [float(x) for x in r]


def query_rows(lat: float, lon: float, elevation_m: float, days: list[dt.date], daily: pd.DataFrame,
               clim_curve: np.ndarray | None) -> pd.DataFrame:
    """Feature rows for every species x day at a point (same feature code as training)."""
    pts = pd.DataFrame(
        [{"taxon_id": s["taxon_id"], "date": d, "lat": lat, "lon": lon, "elevation_m": elevation_m}
         for s in SPECIES for d in days]
    )
    rows = features.feature_rows(pts, daily, clim_curve)
    rows["observed_on"] = rows["date"]
    rows["cell_id"] = weather.cell_id(lat, lon)
    return rows


def forecast(lat: float, lon: float, place_name: str | None = None, today: dt.date | None = None, *,
             classifier=None, weather_daily: pd.DataFrame | None = None, elevation_m: float | None = None,
             context: pd.DataFrame | None = None, clim_curve: np.ndarray | None = None,
             model_config: dict | None = None) -> dict:
    """See docs/CONTRACT.md. Keyword-only arguments allow injecting a classifier (any object with fit,
    predict_proba and classes_), weather, elevation, context pool and climatology (for tests)."""
    t_start = time.time()
    lat, lon = float(lat), float(lon)
    today = today or weather.local_today()
    days = [today + dt.timedelta(days=i) for i in range(N_DAYS)]

    if elevation_m is None:
        elevation_m = weather.elevations([(lat, lon)])[0]
    elevation_m = float(elevation_m) if elevation_m is not None else float("nan")
    clim_table = _climatology() if (weather_daily is None or clim_curve is None) else None
    clat, clon, cid, _known = dataset.weather_point(lat, lon, clim_table)
    wx_meta = {"archive": "not used (weather injected)"}
    if weather_daily is None:
        weather_daily, wx_meta = weather.app_daily(clat, clon, today, PAST_DAYS, FORECAST_DAYS)
    clim_source = "injected"
    if clim_curve is None:
        clim_curve, clim_source = climatology_curve_for(lat, lon, clim_table)

    rows = query_rows(lat, lon, elevation_m, days, weather_daily, clim_curve)
    pool = context if context is not None else _context_pool()
    cfg = dict(LOCKED_CONFIG, **(model_config or {}))
    factory = (lambda: classifier) if classifier is not None else None
    model = PeakweekModel(clf_factory=factory, **cfg).fit(pool)
    t_model = time.time()
    P = model.predict_proba(rows)
    model_seconds = time.time() - t_model

    inside = config.in_region(lat, lon)
    note = config.REGION_NOTE
    if not inside:
        note += " This location is outside that box, so these probabilities are an extrapolation."
    ctx_rows = sorted({len(c) for _, c, _ in model.contexts(rows)}) if hasattr(model, "pool") else []

    species_out = []
    for s in SPECIES:
        daily = []
        for d in days:
            i = int(np.where((rows["taxon_id"] == s["taxon_id"]).to_numpy()
                             & (rows["date"].map(lambda x: pd.Timestamp(x).date()) == d).to_numpy())[0][0])
            g, c, b = _round_probs(P[i])
            daily.append({"date": d.isoformat(), "green": g, "colored": c, "bare": b})
        species_out.append({"taxon_id": s["taxon_id"], "common": s["common"], "scientific": s["scientific"],
                            "daily": daily})

    return {
        "place": {"name": place_name, "lat": round(lat, 4), "lon": round(lon, 4),
                  "elevation_m": None if np.isnan(elevation_m) else round(elevation_m, 1)},
        "in_region": bool(inside),
        "region_note": note,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "today": today.isoformat(),
        "days": [d.isoformat() for d in days],
        "model": {
            "name": TABPFN_NAME if classifier is None else f"injected {type(classifier).__name__}",
            "context_rows": int(max(ctx_rows)) if ctx_rows else int(cfg["n_context"]),
            "seconds": round(model_seconds, 1),
            "wall_seconds": round(time.time() - t_start, 1),
            "training_data": "iNaturalist Leaves annotations, 2018-2025",
            "strategy": cfg["strategy"], "n_estimators": cfg["n_estimators"], "n_subsamples": cfg["n_subsamples"],
            "feature_set": cfg["feature_set"],
            "attribution": "Built with PriorLabs-TabPFN",
        },
        "weather": {"source": "Open-Meteo forecast API (past days from the Open-Meteo ERA5 archive where available)",
                    "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS, **wx_meta,
                    "grid_cell": {"cell_id": cid, "deg": config.GRID_DEG, "weather_lat": clat, "weather_lon": clon},
                    "climatology": clim_source,
                    "attribution": "Weather data by Open-Meteo.com (CC BY 4.0)"},
        "species": species_out,
    }
