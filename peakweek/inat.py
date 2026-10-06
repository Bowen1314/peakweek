"""iNaturalist API v1 fetch (keyless) and Leaves-annotation parsing.

Etiquette: <= 1 request/second, descriptive User-Agent, per_page=200, pagination by id_above
(deep page numbers are capped by the API). Pages are cached on disk (slimmed to the fields we use).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from . import config

API_URL = "https://api.inaturalist.org/v1/observations"
MIN_INTERVAL_S = 1.05
MAX_POSITIONAL_ACCURACY_M = 1000

_last_request = 0.0

KEEP_FIELDS = (
    "id", "observed_on", "location", "obscured", "geoprivacy", "taxon_geoprivacy",
    "positional_accuracy", "public_positional_accuracy", "quality_grade", "captive",
    "license_code", "time_observed_at",
)


def _slim(obs: dict) -> dict:
    out = {k: obs.get(k) for k in KEEP_FIELDS}
    out["taxon_id_returned"] = (obs.get("taxon") or {}).get("id")
    out["annotations"] = [
        {
            "controlled_attribute_id": a.get("controlled_attribute_id"),
            "controlled_value_id": a.get("controlled_value_id"),
            "vote_score": a.get("vote_score", 0),
        }
        for a in (obs.get("annotations") or [])
    ]
    return out


def _http_get_json(url: str, retries: int = 5) -> dict:
    global _last_request
    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def iter_observations(params: dict, cache_dir: Path | None = None, use_cache: bool = True) -> Iterator[dict]:
    """Yield slimmed observations for a query, paginating with id_above (ascending id)."""
    cache_dir = Path(cache_dir or config.CACHE_DIR / "inat")
    base = {k: v for k, v in params.items() if k not in ("id_above", "page", "order", "order_by", "per_page")}
    key = hashlib.sha1(json.dumps(base, sort_keys=True, default=str).encode()).hexdigest()[:16]
    qdir = cache_dir / key
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "query.json").write_text(json.dumps(base, sort_keys=True, default=str, indent=1))
    id_above = 0
    while True:
        path = qdir / f"page_{id_above}.json.gz"
        if use_cache and path.exists():
            with gzip.open(path, "rt") as f:
                page = json.load(f)
        else:
            q = dict(base, order_by="id", order="asc", per_page=200, id_above=id_above)
            raw = _http_get_json(API_URL + "?" + urllib.parse.urlencode(q))
            page = {"total_results": raw.get("total_results"), "results": [_slim(o) for o in raw.get("results", [])]}
            with gzip.open(path, "wt") as f:
                json.dump(page, f)
        results = page["results"]
        if not results:
            return
        yield from results
        if len(results) < 200:
            return
        id_above = max(o["id"] for o in results)


# ---------------------------------------------------------------- parsing


def parse_leaves_label(annotations: Iterable[dict]) -> tuple[str | None, str]:
    """Return (label, reason) from an observation's annotations.

    Only Leaves annotations (attribute 36) with vote_score >= 0 count. Values 38/39/40 map to
    green/colored/bare; 37 (breaking leaf buds) is ignored. If the counted annotations disagree
    the observation is dropped as a conflict.
    """
    values = set()
    any_leaves = False
    for a in annotations or []:
        if a.get("controlled_attribute_id") != config.INAT_LEAVES_TERM:
            continue
        any_leaves = True
        score = a.get("vote_score")
        if score is None:
            score = 0
        if score < 0:
            continue
        v = a.get("controlled_value_id")
        if v in config.INAT_VALUE_TO_LABEL:
            values.add(config.INAT_VALUE_TO_LABEL[v])
    if not any_leaves:
        return None, "no_leaves_annotation"
    if not values:
        return None, "no_usable_leaves_value"  # only 37 and/or downvoted annotations
    if len(values) > 1:
        return None, "conflicting_leaves"
    return values.pop(), "ok"


def parse_location(loc) -> tuple[float, float] | None:
    if not loc:
        return None
    try:
        lat_s, lon_s = str(loc).split(",")
        return float(lat_s), float(lon_s)
    except ValueError:
        return None


def coordinate_problem(obs: dict, region: dict | None = None) -> str | None:
    """Return a drop reason if the observation's coordinates are unusable, else None."""
    if obs.get("obscured"):
        return "obscured"
    if obs.get("geoprivacy") in ("obscured", "private"):
        return "obscured"
    if obs.get("taxon_geoprivacy") in ("obscured", "private"):
        return "obscured"
    ll = parse_location(obs.get("location"))
    if ll is None:
        return "no_location"
    acc = obs.get("positional_accuracy")
    if acc is not None and acc > MAX_POSITIONAL_ACCURACY_M:
        return "positional_accuracy_gt_1000m"
    lat, lon = ll
    r = region or config.REGION
    if not (r["lat_min"] <= lat <= r["lat_max"] and r["lon_min"] <= lon <= r["lon_max"]):
        return "outside_region"
    return None


def quality_problem(obs: dict) -> str | None:
    """Keep verifiable observations only (research or needs_id); casual/captive are dropped."""
    if obs.get("captive"):
        return "captive_or_cultivated"
    if obs.get("quality_grade") not in ("research", "needs_id"):
        return "casual"
    if not obs.get("observed_on"):
        return "no_date"
    return None


def to_record(obs: dict, taxon_id: int, region: dict | None = None) -> tuple[dict | None, str]:
    reason = quality_problem(obs) or coordinate_problem(obs, region)
    if reason:
        return None, reason
    label, why = parse_leaves_label(obs.get("annotations"))
    if label is None:
        return None, why
    lat, lon = parse_location(obs["location"])
    return {
        "obs_id": obs["id"],
        "observed_on": obs["observed_on"],
        "lat": round(lat, 3),
        "lon": round(lon, 3),
        "taxon_id": taxon_id,
        "label": label,
        "quality_grade": obs.get("quality_grade"),
        "captive": bool(obs.get("captive")),
        "license_code": obs.get("license_code"),
    }, "ok"


def fetch_labeled(taxon_ids: Iterable[int], d1: str, d2: str, months: str | None = "9,10,11",
                  region: dict | None = None, use_cache: bool = True, log=print) -> tuple[list[dict], dict]:
    """Fetch all Leaves-annotated observations for the species and return (records, drop_counts).

    Each species is queried once with term_id=36 and no term value; labels are parsed client-side.
    The *queried* species taxon id is stored (the API also returns descendant taxa).
    """
    r = region or config.REGION
    records: list[dict] = []
    drops: dict = {}
    seen: set = set()
    for tid in taxon_ids:
        params = {
            "taxon_id": tid, "term_id": config.INAT_LEAVES_TERM, "d1": d1, "d2": d2,
            "nelat": r["lat_max"], "nelng": r["lon_max"], "swlat": r["lat_min"], "swlng": r["lon_min"],
        }
        if months:
            params["month"] = months
        c = Counter()
        n = 0
        for obs in iter_observations(params, use_cache=use_cache):
            n += 1
            if obs["id"] in seen:
                c["duplicate"] += 1
                continue
            seen.add(obs["id"])
            rec, why = to_record(obs, tid, r)
            c[why] += 1
            if rec:
                records.append(rec)
        drops[tid] = dict(c)
        log(f"taxon {tid}: fetched {n}, kept {c['ok']}, drops {dict((k, v) for k, v in c.items() if k != 'ok')}")
    return records, drops
