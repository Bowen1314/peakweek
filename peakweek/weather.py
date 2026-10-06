"""Open-Meteo (keyless, CC BY 4.0) archive / forecast / elevation fetch with an on-disk cache.

Weather data by Open-Meteo.com (https://open-meteo.com/), licensed CC BY 4.0. The archive API serves
ERA5 / ERA5-Land reanalysis (Copernicus Climate Change Service).

Open-Meteo weights calls: a request for one location counts max(1, n_vars/10) * max(1, n_days/14) calls.
Every network request is appended to .cache/openmeteo/usage.jsonl with its estimated weight, and requests
are throttled to stay under HOURLY_WEIGHT_CAP weighted calls in any rolling hour.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

DAILY_VARS = ("temperature_2m_mean", "temperature_2m_min", "precipitation_sum")
COLUMN_NAMES = {"temperature_2m_mean": "tmean", "temperature_2m_min": "tmin", "precipitation_sum": "prcp"}

HOURLY_WEIGHT_CAP = 4500.0
MINUTE_WEIGHT_CAP = 500.0


class DailyLimitExceeded(RuntimeError):
    pass


def weighted_cost(n_days: int, n_vars: int = len(DAILY_VARS), n_locations: int = 1) -> float:
    """Open-Meteo's documented call weighting (per location; >10 variables or >14 days count extra)."""
    return n_locations * max(1.0, n_vars / 10.0) * max(1.0, n_days / 14.0)


def _cache_dir() -> Path:
    d = config.CACHE_DIR / "openmeteo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _usage_path() -> Path:
    return _cache_dir() / "usage.jsonl"


def usage_log() -> list[dict]:
    p = _usage_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def total_weighted_calls(since: float | None = None) -> float:
    return sum(r["weight"] for r in usage_log() if since is None or r["t"] >= since)


def _throttle(weight: float) -> None:
    while True:
        now = time.time()
        log = [r for r in usage_log() if r["t"] >= now - 3600]
        hour = sum(r["weight"] for r in log)
        minute = sum(r["weight"] for r in log if r["t"] >= now - 60)
        if hour + weight <= HOURLY_WEIGHT_CAP and minute + weight <= MINUTE_WEIGHT_CAP:
            return
        time.sleep(5)


def _record_usage(api: str, weight: float, note: str) -> None:
    with open(_usage_path(), "a") as f:
        f.write(json.dumps({"t": time.time(), "api": api, "weight": round(weight, 3), "note": note}) + "\n")


class OfflineError(Exception):
    """Raised on a cache miss when PEAKWEEK_OFFLINE=1 (deliberately not a RuntimeError / OSError,
    so callers that fall back on network errors do not swallow it)."""


def _check_online(what: str) -> None:
    if os.environ.get("PEAKWEEK_OFFLINE") == "1":
        raise OfflineError(f"PEAKWEEK_OFFLINE=1 and not cached: {what}")


def _get_json_cached(url: str, params: dict, cache_path: Path, weight: float, api: str, retries: int = 6) -> dict:
    """Return cached JSON if present; otherwise fetch, write the raw response to disk, then parse."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    _check_online(f"{api} -> {cache_path}")
    full = url + "?" + urllib.parse.urlencode(params, safe=",")
    for attempt in range(retries):
        _throttle(weight)
        req = urllib.request.Request(full, headers={"User-Agent": config.USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read()
            _record_usage(api, weight, cache_path.name)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_bytes(raw)
            tmp.replace(cache_path)
            return json.loads(raw)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            _record_usage(api, weight, f"HTTP {e.code} {cache_path.name}")
            if e.code == 429:
                if "daily" in body.lower():
                    raise DailyLimitExceeded(body) from e
                time.sleep(60 * (attempt + 1))
                continue
            if e.code >= 500 and attempt < retries - 1:
                time.sleep(15 * (attempt + 1))
                continue
            raise RuntimeError(f"Open-Meteo HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(15 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Open-Meteo request failed after {retries} attempts: {cache_path.name}")


def _daily_frame(js: dict) -> pd.DataFrame:
    d = js["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"])})
    for k, name in COLUMN_NAMES.items():
        df[name] = pd.to_numeric(pd.Series(d.get(k, [None] * len(df))), errors="coerce").astype(float)
    df.attrs["elevation"] = js.get("elevation")
    return df


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def cell_center(lat: float, lon: float, deg: float = config.GRID_DEG) -> tuple[float, float]:
    """Center of the grid cell containing (lat, lon)."""
    clat = math.floor(lat / deg) * deg + deg / 2
    clon = math.floor(lon / deg) * deg + deg / 2
    return round(clat, 4), round(clon, 4)


def cell_id(lat: float, lon: float, deg: float = config.GRID_DEG) -> str:
    clat, clon = cell_center(lat, lon, deg)
    return f"{clat:.2f}_{clon:.2f}"


def season_window(year: int) -> tuple[dt.date, dt.date]:
    """The weather window fetched per cell-year for training: Sep 1 .. Nov 30."""
    return dt.date(year, *config.SEASON_START), dt.date(year, 11, 30)


def archive_daily(lat: float, lon: float, start: dt.date, end: dt.date) -> pd.DataFrame:
    """ERA5 daily weather (local-time aggregation) from the Open-Meteo archive API, cached on disk."""
    params = {
        "latitude": _fmt(lat), "longitude": _fmt(lon),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS), "timezone": config.TIMEZONE,
    }
    n_days = (end - start).days + 1
    path = _cache_dir() / "archive" / f"{_fmt(lat)}_{_fmt(lon)}_{start}_{end}.json"
    js = _get_json_cached(ARCHIVE_URL, params, path, weighted_cost(n_days), "archive")
    return _daily_frame(js)


def forecast_daily(lat: float, lon: float, past_days: int = 92, forecast_days: int = 16,
                   run_date: dt.date | None = None) -> pd.DataFrame:
    """Forecast-API daily weather incl. past_days of recent history. Cached per local run date."""
    run_date = run_date or dt.datetime.now(dt.timezone.utc).astimezone(_tz()).date()
    params = {
        "latitude": _fmt(lat), "longitude": _fmt(lon), "past_days": past_days,
        "forecast_days": forecast_days, "daily": ",".join(DAILY_VARS), "timezone": config.TIMEZONE,
    }
    path = _cache_dir() / "forecast" / f"{_fmt(lat)}_{_fmt(lon)}_{past_days}_{forecast_days}_{run_date}.json"
    js = _get_json_cached(FORECAST_URL, params, path, weighted_cost(past_days + forecast_days), "forecast")
    return _daily_frame(js)


def merge_history(archive: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Daily frame preferring archive (ERA5) values; forecast-API values fill days/fields the archive lacks."""
    a = archive.copy()
    f = forecast.copy()
    a["date"] = pd.to_datetime(a["date"])
    f["date"] = pd.to_datetime(f["date"])
    out = a.set_index("date").combine_first(f.set_index("date"))
    return out[[c for c in ("tmean", "tmin", "prcp") if c in out]].sort_index().reset_index()


def app_daily(lat: float, lon: float, today: dt.date, past_days: int = 92, forecast_days: int = 16) -> tuple[pd.DataFrame, dict]:
    """Weather for app rows: archive API from Sep 1 (season start) through yesterday, forecast API
    (past_days + forecast_days) for today onward and for any day the archive lacks.

    Training rows use the archive (ERA5). In Sep 2026 the forecast API's past_days precipitation was
    ~2.3 mm/day lower and Tmin ~0.4 C lower than ERA5 at five training weather points
    (eval/weather_shift.py), so the past part of the season comes from the archive as in training."""
    fc = forecast_daily(lat, lon, past_days, forecast_days, run_date=today)
    season = dt.date(today.year, *config.SEASON_START)
    start = season if today > season else today - dt.timedelta(days=30)
    end = today - dt.timedelta(days=1)
    try:
        ar = archive_daily(lat, lon, start, end)
    except (RuntimeError, OSError) as e:  # archive unavailable: fall back to forecast-API history only
        return fc, {"archive": f"unavailable ({e.__class__.__name__})"}
    have = ar.dropna(subset=["tmean", "tmin", "prcp"], how="all")
    meta = {"archive_from": str(start), "archive_through": str(have["date"].max().date()) if len(have) else None}
    return merge_history(ar, fc), meta


def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(config.TIMEZONE)
    except Exception:  # pragma: no cover
        return dt.timezone(dt.timedelta(hours=-4))


def local_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).astimezone(_tz()).date()


def elevations(coords: Sequence[tuple[float, float]], decimals: int = 2) -> list[float | None]:
    """Elevation (m, 90 m DEM) for each coordinate, rounded to `decimals` for de-duplication.

    Batches up to 100 coordinates per request and keeps a JSON cache of every looked-up point.
    """
    cpath = _cache_dir() / f"elevation_{decimals}dp.json"
    cache: dict = json.loads(cpath.read_text()) if cpath.exists() else {}
    keys = [f"{round(a, decimals):.{decimals}f},{round(b, decimals):.{decimals}f}" for a, b in coords]
    todo = sorted({k for k in keys if k not in cache})
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        params = {"latitude": ",".join(k.split(",")[0] for k in batch),
                  "longitude": ",".join(k.split(",")[1] for k in batch)}
        h = hashlib.sha1(",".join(batch).encode()).hexdigest()[:16]
        path = _cache_dir() / "elevation" / f"{h}.json"
        # Elevation weighting is undocumented, but observed: each coordinate counts as one call
        # (6 requests x 100 coordinates hit the 600/min limit).
        js = _get_json_cached(ELEVATION_URL, params, path, float(len(batch)), f"elevation[{len(batch)}]")
        for k, e in zip(batch, js["elevation"]):
            cache[k] = e
        cpath.write_text(json.dumps(cache))
    return [cache.get(k) for k in keys]


OPENTOPODATA_URL = "https://api.opentopodata.org/v1/srtm90m"
_last_otd = 0.0


def elevations_srtm(coords: Sequence[tuple[float, float]], decimals: int = 3) -> list[float | None]:
    """Elevation (m) from the keyless OpenTopoData public API, SRTM 90 m dataset (public domain).

    Used for the training observations: 100 locations per request, <= 1 request/second (public API
    limits: 100 locations/request, 1 call/s, 1000 calls/day). Open-Meteo's elevation endpoint weights
    each coordinate as one call, which would not fit the build's Open-Meteo budget."""
    global _last_otd
    root = config.CACHE_DIR / "opentopodata"
    root.mkdir(parents=True, exist_ok=True)
    cpath = root / f"srtm90m_{decimals}dp.json"
    cache: dict = json.loads(cpath.read_text()) if cpath.exists() else {}
    keys = [f"{round(a, decimals):.{decimals}f},{round(b, decimals):.{decimals}f}" for a, b in coords]
    todo = sorted({k for k in keys if k not in cache})
    for i in range(0, len(todo), 100):
        batch = todo[i:i + 100]
        h = hashlib.sha1("|".join(batch).encode()).hexdigest()[:16]
        path = root / f"{h}.json"
        if path.exists():
            js = json.loads(path.read_text())
        else:
            _check_online(f"opentopodata {len(batch)} points")
            for attempt in range(5):
                wait = 1.1 - (time.time() - _last_otd)
                if wait > 0:
                    time.sleep(wait)
                _last_otd = time.time()
                url = OPENTOPODATA_URL + "?locations=" + urllib.parse.quote("|".join(batch), safe=",|")
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
                    with urllib.request.urlopen(req, timeout=90) as resp:
                        raw = resp.read()
                    path.write_bytes(raw)
                    js = json.loads(raw)
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                    if attempt == 4:
                        raise
                    time.sleep(10 * (attempt + 1))
        for k, r in zip(batch, js["results"]):
            cache[k] = r.get("elevation")
        cpath.write_text(json.dumps(cache))
    return [cache.get(k) for k in keys]


def nearest_complete(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Reindex a daily frame to a complete date range [start, end] (missing days become NaN rows)."""
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    return df.set_index("date").reindex(idx).rename_axis("date").reset_index()


def usage_summary() -> dict:
    log = usage_log()
    by_api: dict = {}
    for r in log:
        api = r["api"].split("[")[0]
        by_api.setdefault(api, {"requests": 0, "weight": 0.0})
        by_api[api]["requests"] += 1
        by_api[api]["weight"] += r["weight"]
    return {"total_weight": round(sum(r["weight"] for r in log), 1), "requests": len(log), "by_api": by_api}


def as_array(values: Iterable) -> np.ndarray:
    return np.asarray(list(values), dtype=float)
