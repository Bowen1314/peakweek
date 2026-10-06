"""Turn labeled iNaturalist observations + Open-Meteo weather into model-ready feature rows."""

from __future__ import annotations

import datetime as dt
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import config, features, weather

OBS_COLUMNS = ["obs_id", "observed_on", "lat", "lon", "taxon_id", "label", "quality_grade", "captive", "license_code"]
FEATURE_TABLE_COLUMNS = (
    ["obs_id", "observed_on", "year", "taxon_id", "label", "label_int", "cell_id"]
    + features.FEATURES_FULL + ["tmean_season"]
)

WeatherLoader = Callable[[float, float, int], pd.DataFrame]


def observations_frame(records: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(records), columns=OBS_COLUMNS)
    df = df.sort_values(["observed_on", "obs_id"]).reset_index(drop=True)
    return df


def license_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep observations with a non-null license_code (these may be redistributed under their license)."""
    keep = df["license_code"].notna() & (df["license_code"].astype(str) != "")
    return df[keep].reset_index(drop=True), int((~keep).sum())


def assign_cells(df: pd.DataFrame, deg: float = config.GRID_DEG) -> pd.DataFrame:
    """Add cell_id (named by the geometric cell center), the cell's weather point (cell_lat, cell_lon =
    mean coordinate of the cell's observations in df, rounded to 0.01 deg) and year."""
    out = df.copy()
    out["cell_id"] = [weather.cell_id(a, b, deg) for a, b in zip(out["lat"], out["lon"])]
    wp = out.groupby("cell_id")[["lat", "lon"]].mean().round(2)
    out["cell_lat"] = out["cell_id"].map(wp["lat"])
    out["cell_lon"] = out["cell_id"].map(wp["lon"])
    out["year"] = pd.to_datetime(out["observed_on"]).dt.year
    return out


def weather_point(lat: float, lon: float, clim: pd.DataFrame | None) -> tuple[float, float, str, bool]:
    """(wx_lat, wx_lon, cell_id, known) for a point: the training cell's weather point if the cell is in
    the climatology table, else the geometric cell center."""
    cid = weather.cell_id(lat, lon)
    if clim is not None:
        sub = clim[clim["cell_id"] == cid]
        if not sub.empty:
            return float(sub["cell_lat"].iloc[0]), float(sub["cell_lon"].iloc[0]), cid, True
    clat, clon = weather.cell_center(lat, lon)
    return clat, clon, cid, False


def weather_plan(df_cells: pd.DataFrame, years: Iterable[int] = config.TRAIN_YEARS) -> dict:
    """Cell-years to fetch (every occupied cell x every year, for the leave-one-year-out climatology)."""
    cells = df_cells[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates().sort_values("cell_id")
    years = list(years)
    n_days = (weather.season_window(2021)[1] - weather.season_window(2021)[0]).days + 1
    per = weather.weighted_cost(n_days)
    occupied = df_cells[["cell_id", "year"]].drop_duplicates()
    return {
        "n_cells": len(cells), "years": years, "n_cell_years": len(cells) * len(years),
        "n_occupied_cell_years": len(occupied), "weight_per_cell_year": per,
        "weight_full": round(len(cells) * len(years) * per, 1),
        "weight_occupied_only": round(len(occupied) * per, 1),
        "cells": cells,
    }


def archive_loader(clat: float, clon: float, year: int) -> pd.DataFrame:
    start, end = weather.season_window(year)
    return weather.archive_daily(clat, clon, start, end)


def cell_curves(clat: float, clon: float, years: Iterable[int], loader: WeatherLoader = archive_loader) -> dict:
    """{year: tmean_season curve} for one cell."""
    out = {}
    for y in years:
        prepared = features.prepare_daily(loader(clat, clon, y))
        out[y] = features.season_mean_curve(prepared, y)
    return out


def climatology_rows(cell_id: str, clat: float, clon: float, curves: dict) -> pd.DataFrame:
    total, count = features.climatology_from_curves(list(curves.values()))
    mean = features.leave_one_out_mean(total, count, None)
    return pd.DataFrame({
        "cell_id": cell_id, "cell_lat": clat, "cell_lon": clon,
        "offset": np.arange(features.N_SEASON_OFFSETS),
        "clim_tmean_season": np.round(mean, 4), "n_years": count.astype(int),
    })


def build_feature_table(obs: pd.DataFrame, elevation: dict, loader: WeatherLoader = archive_loader,
                        years: Iterable[int] = config.TRAIN_YEARS, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Feature rows for every observation (weather from its cell-year, anomaly vs leave-one-year-out
    climatology of the same cell) and the all-years climatology table used by the app.

    obs must have cell columns (assign_cells); elevation maps obs_id -> metres."""
    years = list(years)
    parts = []
    clim_parts = []
    cells = obs[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates().sort_values("cell_id")
    for i, (cid, clat, clon) in enumerate(cells.itertuples(index=False)):
        curves = cell_curves(clat, clon, years, loader)
        clim_parts.append(climatology_rows(cid, clat, clon, curves))
        total, count = features.climatology_from_curves(list(curves.values()))
        sub = obs[obs["cell_id"] == cid]
        for y, grp in sub.groupby("year"):
            daily = loader(clat, clon, int(y))
            loo = features.leave_one_out_mean(total, count, curves.get(int(y)))
            pts = grp.rename(columns={"observed_on": "date"}).copy()
            pts["elevation_m"] = pts["obs_id"].map(elevation).astype(float)
            fr = features.feature_rows(pts, daily, loo)
            fr = fr.rename(columns={"date": "observed_on"})
            parts.append(fr)
        if (i + 1) % 25 == 0:
            log(f"features: {i + 1}/{len(cells)} cells")
    table = pd.concat(parts, ignore_index=True)
    table["label_int"] = table["label"].map(config.LABEL_TO_INT).astype(int)
    table["year"] = table["year"].astype(int)
    table = table[FEATURE_TABLE_COLUMNS].sort_values(["observed_on", "obs_id"]).reset_index(drop=True)
    clim = pd.concat(clim_parts, ignore_index=True)
    return table, clim


def load_features(path=None) -> pd.DataFrame:
    path = path or config.DATA_DIR / "features.csv"
    df = pd.read_csv(path)
    df["observed_on"] = pd.to_datetime(df["observed_on"]).dt.date
    return df


def load_climatology(path=None) -> pd.DataFrame:
    path = path or config.DATA_DIR / "climatology.csv"
    return pd.read_csv(path)


def climatology_curve(clim: pd.DataFrame, cid: str) -> np.ndarray | None:
    sub = clim[clim["cell_id"] == cid].sort_values("offset")
    if sub.empty:
        return None
    curve = np.full(features.N_SEASON_OFFSETS, np.nan)
    curve[sub["offset"].to_numpy()] = sub["clim_tmean_season"].to_numpy()
    return curve


def split(df: pd.DataFrame, train_years: Iterable[int], eval_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ty = set(train_years)
    return df[df["year"].isin(ty)].reset_index(drop=True), df[df["year"] == eval_year].reset_index(drop=True)


SPLITS = {
    "val": (range(2018, 2024), 2024),
    "test": (range(2018, 2025), 2025),
}


def today_local() -> dt.date:
    return weather.local_today()
