"""Feature computation. Pure functions: the same code builds training rows and app (forecast) rows.

Conventions
- Every seasonal accumulator starts on Sep 1 of the target date's year (config.SEASON_START), so the app's
  forecast call (past_days=92) always covers it.
- Features for target date d use weather from days strictly before d (window ends at d-1).
- Missing days: interior temperature gaps of <= 2 consecutive days are linearly interpolated; the first 2
  days of a trailing gap are forward-filled (e.g. the forecast's incomplete last day); longer gaps stay
  missing. After that, a sum/count accumulator whose window still has a missing day is NaN,
  and a windowed mean is NaN unless at least half of its window is present (NaNs are skipped).
- Precipitation is never interpolated; its windowed mean skips missing days with the same half-window rule.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Sequence

import numpy as np
import pandas as pd

from . import config
from .species import SPECIES_CODE

WEATHER_FEATURES = ["cdd20", "frost0", "frost5", "tmin7", "tmean14", "prcp30", "tanom"]
CALENDAR_FEATURES = ["species_code", "lat", "lon", "elevation_m", "doy"]
FEATURES_FULL = CALENDAR_FEATURES + ["daylength_h"] + WEATHER_FEATURES
CATEGORICAL = ["species_code"]
N_SEASON_OFFSETS = 92  # Sep 1 .. Dec 1 (offsets 0..91); training weather covers Sep 1..Nov 30

CDD_BASE_C = 20.0


# ----------------------------------------------------------------------------- astronomy

def daylength_hours(lat: float, doy: int | float) -> float:
    """Day length in hours (CBM model, Forsythe et al. 1995, p = 0.8333 deg: sunrise/sunset at the
    top of the solar disk with standard refraction)."""
    p = 0.8333
    theta = 0.2163108 + 2 * math.atan(0.9671396 * math.tan(0.00860 * (doy - 186)))
    phi = math.asin(0.39795 * math.cos(theta))
    lat_r = math.radians(lat)
    num = math.sin(math.radians(p)) + math.sin(lat_r) * math.sin(phi)
    den = math.cos(lat_r) * math.cos(phi)
    x = min(1.0, max(-1.0, num / den))
    return 24.0 - (24.0 / math.pi) * math.acos(x)


def season_start(d: dt.date) -> dt.date:
    return dt.date(d.year, *config.SEASON_START)


def season_offset(d: dt.date) -> int:
    """Days since Sep 1 of d's year (negative before Sep 1)."""
    return (d - season_start(d)).days


# ----------------------------------------------------------------------------- weather series

def prepare_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Return a complete-calendar daily frame indexed by date with short temperature gaps filled.

    `daily` has columns date, tmean, tmin, prcp (any of them may contain NaN; dates may be missing)."""
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    if len(df) == 0:
        return df
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    for col in ("tmean", "tmin"):
        if col in df:
            df[col] = fill_short_gaps(df[col].to_numpy(dtype=float), MAX_FILL_GAP)
    df.index.name = "date"
    return df


MAX_FILL_GAP = 2


def fill_short_gaps(x: np.ndarray, max_gap: int = MAX_FILL_GAP) -> np.ndarray:
    """Linearly interpolate interior NaN runs of length <= max_gap; forward-fill the first max_gap days
    of a trailing NaN run. Longer interior runs and leading runs stay NaN."""
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    i = 0
    while i < n:
        if not np.isnan(x[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(x[j]):
            j += 1
        run = j - i
        if i > 0 and j < n and run <= max_gap:  # interior short gap
            a, b = x[i - 1], x[j]
            for k in range(run):
                x[i + k] = a + (b - a) * (k + 1) / (run + 1)
        elif i > 0 and j == n:  # trailing gap
            x[i:min(n, i + max_gap)] = x[i - 1]
        i = j
    return x


def _window(series: pd.Series, start: dt.date, end_incl: dt.date) -> np.ndarray:
    """Values for every calendar day in [start, end_incl]; days outside the series are NaN."""
    if end_incl < start:
        return np.array([], dtype=float)
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end_incl), freq="D")
    return series.reindex(idx).to_numpy(dtype=float)


def _sum_or_nan(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    if np.isnan(x).any():
        return float("nan")
    return float(x.sum())


def _mean_half(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    present = ~np.isnan(x)
    if present.sum() * 2 < x.size:
        return float("nan")
    return float(x[present].mean())


def chill_degree_days(tmean: np.ndarray, base: float = CDD_BASE_C) -> float:
    """Sum of max(0, base - Tmean); NaN if any day is missing."""
    t = np.asarray(tmean, dtype=float)
    return _sum_or_nan(np.maximum(0.0, base - t) if t.size else t)


def count_nights_at_or_below(tmin: np.ndarray, threshold: float) -> float:
    """Number of days with Tmin <= threshold; NaN if any day is missing."""
    t = np.asarray(tmin, dtype=float)
    if t.size and np.isnan(t).any():
        return float("nan")
    return float((t <= threshold).sum())


def weather_features_on(prepared: pd.DataFrame, d: dt.date) -> dict:
    """Seasonal weather features for target date d from a prepared daily frame (see prepare_daily).

    tanom is not computed here (it needs a climatology); tmean_season is returned for it."""
    s = season_start(d)
    last = d - dt.timedelta(days=1)
    nan = float("nan")
    if d < s:
        return {k: nan for k in ("cdd20", "frost0", "frost5", "tmin7", "tmean14", "prcp30", "tmean_season")}
    tmean = prepared["tmean"] if "tmean" in prepared else pd.Series(dtype=float)
    tmin = prepared["tmin"] if "tmin" in prepared else pd.Series(dtype=float)
    prcp = prepared["prcp"] if "prcp" in prepared else pd.Series(dtype=float)

    def lb(days: int) -> dt.date:
        return max(s, d - dt.timedelta(days=days))

    season_tmean = _window(tmean, s, last)
    season_tmin = _window(tmin, s, last)
    return {
        "cdd20": chill_degree_days(season_tmean),
        "frost0": count_nights_at_or_below(season_tmin, 0.0),
        "frost5": count_nights_at_or_below(season_tmin, 5.0),
        "tmin7": _mean_half(_window(tmin, lb(7), last)),
        "tmean14": _mean_half(_window(tmean, lb(14), last)),
        "prcp30": _mean_half(_window(prcp, lb(30), last)),
        "tmean_season": _mean_half(season_tmean),
    }


def season_mean_curve(prepared: pd.DataFrame, year: int) -> np.ndarray:
    """tmean_season (mean Tmean over [Sep 1, d-1]) for d = Sep 1 + k, k = 0..N_SEASON_OFFSETS-1."""
    s = dt.date(year, *config.SEASON_START)
    out = np.full(N_SEASON_OFFSETS, np.nan)
    tmean = prepared["tmean"] if "tmean" in prepared else pd.Series(dtype=float)
    vals = _window(tmean, s, s + dt.timedelta(days=N_SEASON_OFFSETS - 2))
    for k in range(1, N_SEASON_OFFSETS):
        out[k] = _mean_half(vals[:k])
    return out


def climatology_from_curves(curves: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-offset (sum, count) over years, skipping NaN. Mean = sum / count."""
    arr = np.vstack(curves) if len(curves) else np.full((0, N_SEASON_OFFSETS), np.nan)
    present = ~np.isnan(arr)
    return np.where(present, arr, 0.0).sum(axis=0), present.sum(axis=0).astype(float)


def leave_one_out_mean(total: np.ndarray, count: np.ndarray, own: np.ndarray | None) -> np.ndarray:
    """Climatological mean excluding one year's curve (`own`); with own=None, the plain mean."""
    total = np.asarray(total, dtype=float)
    count = np.asarray(count, dtype=float)
    if own is not None:
        own = np.asarray(own, dtype=float)
        has = ~np.isnan(own)
        total = total - np.where(has, own, 0.0)
        count = count - has
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.where(count > 0, count, 1), np.nan)


# ----------------------------------------------------------------------------- rows

def feature_rows(points: pd.DataFrame, daily: pd.DataFrame, clim_curve: np.ndarray | None) -> pd.DataFrame:
    """Model-ready feature rows for points that share one weather series.

    points: columns taxon_id, date (date-like), lat, lon, elevation_m (extra columns are kept).
    daily: raw daily weather (date, tmean, tmin, prcp) covering Sep 1 .. max(date)-1.
    clim_curve: climatological tmean_season by season offset (len N_SEASON_OFFSETS) or None.
    """
    prepared = prepare_daily(daily)
    out = points.copy().reset_index(drop=True)
    dates = [pd.Timestamp(x).date() for x in out["date"]]
    cache: dict = {}
    wf = []
    for d in dates:
        if d not in cache:
            cache[d] = weather_features_on(prepared, d)
        wf.append(cache[d])
    wfd = pd.DataFrame(wf)
    out["species_code"] = out["taxon_id"].map(SPECIES_CODE).astype(float)
    out["doy"] = [d.timetuple().tm_yday for d in dates]
    out["daylength_h"] = [daylength_hours(la, doy) for la, doy in zip(out["lat"], out["doy"])]
    for k in ("cdd20", "frost0", "frost5", "tmin7", "tmean14", "prcp30", "tmean_season"):
        out[k] = wfd[k].to_numpy()
    offs = np.array([season_offset(d) for d in dates])
    if clim_curve is None:
        clim = np.full(len(out), np.nan)
    else:
        clim_curve = np.asarray(clim_curve, dtype=float)
        ok = (offs >= 0) & (offs < len(clim_curve))
        clim = np.where(ok, clim_curve[np.clip(offs, 0, len(clim_curve) - 1)], np.nan)
    out["tanom"] = out["tmean_season"].to_numpy() - clim
    return out
