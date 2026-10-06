"""Are these trees actually recorded near a place? One keyless iNaturalist call, cached on disk.

    counts(40.49, -74.45) -> {48098: 2721, 52543: 770, ...}   (None if iNaturalist could not be asked)

Counts are research-grade, verifiable iNaturalist observations within RADIUS_KM of the point, per
species (observations of a variety or subspecies count toward its species). The point is rounded
to 0.01 degree (about 1 km) for both the query and the cache key.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

API_URL = "https://api.inaturalist.org/v1/observations/species_counts"
USER_AGENT = "peakweek/0.1 (hackathon project; contact via GitHub)"
RADIUS_KM = 50
SOURCE = "iNaturalist research-grade observations"
CACHE_TTL = 30 * 24 * 3600
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = Path(os.environ.get("PEAKWEEK_CACHE_DIR", str(ROOT / ".cache"))) / "nearby"

# The 8 forecast species (same ids as peakweek/species.py, which needs nothing outside the stdlib
# but is the core's file; kept here as a fallback so this module stands alone).
_FALLBACK_IDS = (48098, 52543, 49658, 49005, 49202, 54802, 54795, 54763)

_lock = threading.Lock()


def default_taxon_ids():
    try:
        from .species import TAXON_IDS
        return list(TAXON_IDS)
    except ImportError:
        return list(_FALLBACK_IDS)


def _round(x: float) -> float:
    return round(float(x), 2) + 0.0  # + 0.0 turns -0.0 into 0.0


def cache_key(lat: float, lon: float) -> str:
    """'40.4862, -74.4518' -> '40.49_-74.45' (also the cache file name)."""
    return "%.2f_%.2f" % (_round(lat), _round(lon))


def query_url(lat: float, lon: float, taxon_ids: Iterable[int]) -> str:
    return API_URL + "?" + urllib.parse.urlencode({
        "lat": "%.2f" % _round(lat), "lng": "%.2f" % _round(lon), "radius": RADIUS_KM,
        "taxon_id": ",".join(str(int(t)) for t in taxon_ids),
        "quality_grade": "research", "verifiable": "true",
    })


def parse_response(data: dict, taxon_ids: Iterable[int]) -> Dict[int, int]:
    """species_counts JSON -> {taxon_id: count} for exactly `taxon_ids` (missing ones are 0)."""
    wanted = [int(t) for t in taxon_ids]
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("unexpected iNaturalist response")
    out = {t: 0 for t in wanted}
    for row in data["results"]:
        taxon = row.get("taxon") or {}
        lineage = [taxon.get("id")] + list(taxon.get("ancestor_ids") or [])
        for t in wanted:
            if t in lineage:
                out[t] += int(row.get("count") or 0)
                break
    return out


def _fetch(lat: float, lon: float, taxon_ids, timeout: float, opener: Callable) -> dict:
    req = urllib.request.Request(query_url(lat, lon, taxon_ids), headers={"User-Agent": USER_AGENT})
    with opener(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _read_cache(path: Path, taxon_ids) -> Optional[Dict[int, int]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - float(data["t"]) >= CACHE_TTL:
            return None
        got = {int(k): int(v) for k, v in data["counts"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if not all(t in got for t in taxon_ids):
        return None  # cached for a different species list
    return {t: got[t] for t in taxon_ids}


def _write_cache(path: Path, counts: Dict[int, int]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"t": time.time(), "counts": {str(k): v for k, v in counts.items()}}, f)
        os.replace(tmp, path)
    except OSError:
        pass  # the cache is a convenience only


def counts(lat: float, lon: float, taxon_ids: Optional[Iterable[int]] = None,
           cache_dir: Optional[Path] = DEFAULT_DIR, timeout: float = 10.0,
           opener: Callable = urllib.request.urlopen) -> Optional[Dict[int, int]]:
    """{taxon_id: research-grade observations within 50 km}, or None if iNaturalist could not be asked.

    `cache_dir=None` disables the cache; `opener` is injectable for tests. Failures are not cached.
    """
    ids = [int(t) for t in (taxon_ids if taxon_ids is not None else default_taxon_ids())]
    path = Path(cache_dir) / (cache_key(lat, lon) + ".json") if cache_dir is not None else None
    if path is not None:
        with _lock:
            hit = _read_cache(path, ids)
        if hit is not None:
            return hit
    try:
        result = parse_response(_fetch(lat, lon, ids, timeout, opener), ids)
    except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError, AttributeError):
        return None  # URLError and timeouts are OSErrors; bad JSON is a ValueError
    if path is not None:
        with _lock:
            _write_cache(path, result)
    return result
