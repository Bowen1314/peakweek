"""Place search via the keyless Open-Meteo geocoding API, with a tiny on-disk cache.

    search("New Brunswick, NJ")  -> [{"name", "label", "short", "lat", "lon", ...}, ...]
    parse_latlon("40.49, -74.45") -> (40.49, -74.45)
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Tuple

API_URL = "https://geocoding-api.open-meteo.com/v1/search"
USER_AGENT = "peakweek/0.1 (hackathon project)"
CACHE_TTL = 30 * 24 * 3600
CACHE_MAX = 500
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = Path(os.environ.get("PEAKWEEK_CACHE_DIR", str(ROOT / ".cache"))) / "geocode.json"

US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_LATLON = re.compile(r"^\s*(%s)\s*(?:,|\s)\s*(%s)\s*$" % (_NUM, _NUM))
_lock = threading.Lock()


def parse_latlon(text: str) -> Optional[Tuple[float, float]]:
    """'40.49, -74.45' or '40.49 -74.45' -> (lat, lon). None if it is not a coordinate pair.

    Raises ValueError for a coordinate pair that is out of range.
    """
    m = _LATLON.match(text or "")
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    check_latlon(lat, lon)
    return lat, lon


def check_latlon(lat: float, lon: float) -> None:
    if not (-90.0 <= lat <= 90.0):
        raise ValueError("latitude must be between -90 and 90")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError("longitude must be between -180 and 180")


def latlon_place(lat: float, lon: float) -> dict:
    label = "%.4f, %.4f" % (lat, lon)
    return {"name": label, "label": label, "short": label, "lat": lat, "lon": lon,
            "admin1": None, "country": None, "country_code": None}


def parse_response(data: dict) -> List[dict]:
    """Turn an Open-Meteo geocoding response into small place dicts."""
    out = []
    for r in (data or {}).get("results") or []:
        try:
            lat, lon = float(r["latitude"]), float(r["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        name = str(r.get("name") or "").strip()
        if not name:
            continue
        admin1 = r.get("admin1")
        country = r.get("country")
        cc = r.get("country_code")
        label = ", ".join(x for x in (name, admin1, country) if x)
        if cc == "US" and admin1 in US_STATES:
            short = "%s, %s" % (name, US_STATES[admin1])
        else:
            short = ", ".join(x for x in (name, admin1 or country) if x)
        out.append({"name": name, "label": label, "short": short, "lat": lat, "lon": lon,
                    "admin1": admin1, "country": country, "country_code": cc,
                    "elevation_m": r.get("elevation")})
    return out


def _fetch(query: str, count: int, timeout: float, opener: Callable) -> dict:
    url = API_URL + "?" + urllib.parse.urlencode(
        {"name": query, "count": count, "language": "en", "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_cache(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(path: Path, cache: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(cache) > CACHE_MAX:
            keep = sorted(cache.items(), key=lambda kv: kv[1].get("t", 0))[-CACHE_MAX:]
            cache = dict(keep)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, path)
    except OSError:
        pass  # the cache is a convenience only


def search(query: str, count: int = 5, cache_path: Optional[Path] = DEFAULT_CACHE,
           timeout: float = 10.0, opener: Callable = urllib.request.urlopen) -> List[dict]:
    """Search for a place by name ("New Brunswick, NJ") or accept "lat, lon" directly.

    `cache_path=None` disables the cache; `opener` is injectable for tests.
    """
    query = (query or "").strip()
    if not query:
        return []
    ll = parse_latlon(query)
    if ll is not None:
        return [latlon_place(*ll)]
    key = "%s|%d" % (" ".join(query.lower().split()), count)
    if cache_path is not None:
        with _lock:
            hit = _load_cache(Path(cache_path)).get(key)
        if hit and time.time() - hit.get("t", 0) < CACHE_TTL:
            return hit["results"]
    results = parse_response(_fetch(query, count, timeout, opener))
    if not results and "," in query:  # "Town, ST" did not match; try the town alone
        results = parse_response(_fetch(query.split(",")[0].strip(), count, timeout, opener))
    if cache_path is not None:
        with _lock:
            cache = _load_cache(Path(cache_path))
            cache[key] = {"t": time.time(), "results": results}
            _save_cache(Path(cache_path), cache)
    return results
