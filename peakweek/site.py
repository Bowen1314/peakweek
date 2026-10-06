"""The daily static site: one precomputed forecast per 1-degree grid cell, rebuilt every morning.

scripts/build_site.py runs this on GitHub Actions (.github/workflows/pages.yml); GitHub Pages serves the
result, so the public site needs no server and no model at request time.

Which cells
    Exactly the grid cells (config.GRID_DEG) that contain training observations, i.e. the cells listed in
    data/climatology.csv (103 of the 150 cells that touch the training box). The other 47 are open water
    (Atlantic, Gulf of Maine) or thinly populated land at the edge of the box with no observation at all;
    they have no training weather point, so they get no forecast and the page says so.

Where in the cell
    A cell's forecast is made at its weather point: the mean coordinate of the cell's training observations
    (rounded to 0.01 degree; land by construction). That point is the location feature (lat, lon), its
    elevation (Open-Meteo elevation API) is the elevation feature, and the weather comes from the same point,
    exactly as forecast(lat, lon) does for a point there. The iNaturalist "recorded within 50 km" check is
    for the same point, fetched once during development (data/nearby_cells.json, scripts/build_nearby.py).

How
    Weather for every cell is fetched with multi-location Open-Meteo requests and stored at the same
    per-location cache paths the single-point path reads (weather.prefetch_app_daily). The model is
    fitted once with the locked configuration and predicts every cell's 120 rows in one predict_proba
    call (the shared stratified context does not depend on the query). Each cell's result goes through
    the unchanged verdict.annotate and card.render_html into the same JSON shape the server returns
    from /api/forecast/status when a job is done.

Batch = single point
    Before anything is written, a few cells are forecast again with the original forecast(lat, lon)
    (same point, same today, same cached weather) and their probabilities must match the batch within
    TOLERANCE, with identical verdicts and headline; otherwise the build fails and nothing is published.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import card, config, dataset, forecast, verdict, weather
from .model import TABPFN_NAME, PeakweekModel

NEARBY_FILE = config.DATA_DIR / "nearby_cells.json"
STATIC_DIR = config.ROOT / "static"
STATIC_ASSETS = ("app.js", "style.css")
# Marker the page looks for to switch app.js to static mode (relative to the page, no leading slash).
STATIC_META = '<meta name="peakweek-static" content="data/index.json">'

# Cells re-run through forecast(): first cell, a chunk boundary (16th index), a northern cell, the last cell.
DEFAULT_CHECK = ("38.50_-74.50", "40.50_-74.50", "44.50_-73.50", "47.50_-79.50")
DEFAULT_CHUNK = 1920      # rows per predict_proba call: 16 cells x 120 rows
TOLERANCE = 1e-5          # max |batch - single| per probability (after the usual 6-decimal rounding)
UPDATE_NOTE = "Rebuilt every morning (about 10:20 UTC) by GitHub Actions."
CELL_RULE = ("A 1-degree grid cell has a forecast if the training data has observations in it; the forecast "
             "is made at the cell's weather point (the mean coordinate of those observations).")


# ----------------------------------------------------------------------------- cells

def published_cells(clim: Optional[pd.DataFrame] = None) -> List[dict]:
    """Every cell with training observations: id, weather point and bounds, sorted by id.

    Bounds are half-open [min, max), the same cells weather.cell_id() assigns (floor of lat / GRID_DEG)."""
    clim = dataset.load_climatology() if clim is None else clim
    cells = clim[["cell_id", "cell_lat", "cell_lon"]].drop_duplicates().sort_values("cell_id")
    half = config.GRID_DEG / 2
    out = []
    for cid, lat, lon in cells.itertuples(index=False):
        if weather.cell_id(lat, lon) != cid:
            raise ValueError("weather point %s, %s is not inside cell %s" % (lat, lon, cid))
        clat, clon = weather.cell_center(lat, lon)
        out.append({"id": cid, "lat": float(lat), "lon": float(lon),
                    "lat_min": round(clat - half, 4), "lat_max": round(clat + half, 4),
                    "lon_min": round(clon - half, 4), "lon_max": round(clon + half, 4)})
    return out


def find_cell(cells: Sequence[dict], lat: float, lon: float) -> Optional[dict]:
    """The published cell whose bounds contain (lat, lon), or None (same rule as app.js findCell)."""
    for c in cells:
        if c["lat_min"] <= lat < c["lat_max"] and c["lon_min"] <= lon < c["lon_max"]:
            return c
    return None


def cell_name(cell: dict) -> str:
    """Place name baked into a cell's card; the page replaces it with the name the user searched."""
    return "1° grid cell around %.2f, %.2f" % (cell["lat"], cell["lon"])


def cell_note(cell: dict, today: dt.date) -> str:
    return "Forecast made %s for the 1° grid cell around %.2f, %.2f; updated daily." % (
        verdict.fmt_day(today), cell["lat"], cell["lon"])


def load_nearby(path: Path = NEARBY_FILE) -> Dict[str, Optional[dict]]:
    """{cell_id: {taxon_id (str): count} or None} from data/nearby_cells.json ({} if the file is missing)."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    return {cid: (v.get("counts") if isinstance(v, dict) else None) for cid, v in data["cells"].items()}


# ----------------------------------------------------------------------------- batch forecast

def batch_forecast(cells: Sequence[dict], today: dt.date, *, clf_factory: Optional[Callable] = None,
                   model_name: str = TABPFN_NAME, inject: Optional[Callable[[dict], dict]] = None,
                   context: Optional[pd.DataFrame] = None, chunk: int = DEFAULT_CHUNK, log=print) -> tuple:
    """forecast() results for every cell from ONE fit and ONE predict_proba over all cells' rows.

    inject(cell) may return prepare_point keyword arguments (weather_daily, elevation_m, clim_curve) for
    tests and the --fake build. Returns ({cell_id: result}, stats)."""
    t0 = time.time()
    preps = []
    for cell in cells:
        preps.append(forecast.prepare_point(cell["lat"], cell["lon"], today, **(inject(cell) if inject else {})))
    rows = pd.concat([p["rows"] for p in preps], ignore_index=True)
    t_prep = time.time() - t0
    log("batch: %d cells, %d query rows (features in %.1f s)" % (len(cells), len(rows), t_prep))

    pool = context if context is not None else forecast._context_pool()
    cfg = dict(forecast.LOCKED_CONFIG)
    model = PeakweekModel(clf_factory=clf_factory, chunk=chunk, **cfg).fit(pool)
    t1 = time.time()
    P = model.predict_proba(rows)
    predict_seconds = time.time() - t1
    log("batch: predicted %d rows in %.1f s (chunk %d)" % (len(rows), predict_seconds, chunk))
    ctx = forecast.context_rows_used(model, rows, cfg)

    results, i = {}, 0
    batch_info = {"cells": len(cells), "rows": int(len(rows)), "chunk": int(chunk),
                  "predict_seconds": round(predict_seconds, 1)}
    for cell, prep in zip(cells, preps):
        n = len(prep["rows"])
        r = forecast.result_from(prep, P[i:i + n], cell_name(cell), model_name=model_name, context_rows=ctx,
                                 cfg=cfg, seconds=None, wall_seconds=None)
        r["model"]["batch"] = dict(batch_info)
        results[cell["id"]] = r
        i += n
    stats = dict(batch_info, prepare_seconds=round(t_prep, 1), context_rows=ctx, fits=model.stats["fits"])
    return results, stats


# ----------------------------------------------------------------------------- batch = single point

_VOLATILE_MODEL_KEYS = ("name", "seconds", "wall_seconds", "batch")


def _probs(result: dict) -> np.ndarray:
    return np.array([[d[k] for k in verdict.KEYS] for s in result["species"] for d in s["daily"]], dtype=float)


def _skeleton(result: dict) -> dict:
    """The result without probabilities and run-specific fields, for an exact comparison."""
    r = copy.deepcopy(result)
    r.pop("generated_at", None)
    r.pop("synthetic", None)
    r["place"].pop("name", None)
    for k in _VOLATILE_MODEL_KEYS:
        r["model"].pop(k, None)
    for s in r["species"]:
        for d in s["daily"]:
            for k in verdict.KEYS:
                d[k] = None
    return r


def compare(batch_result: dict, single_result: dict, nearby_counts: Optional[dict] = None) -> dict:
    """How a batch result differs from forecast() for the same cell."""
    a, b = _probs(batch_result), _probs(single_result)
    same_shape = a.shape == b.shape
    max_diff = float(np.max(np.abs(a - b))) if same_shape and a.size else float("inf")
    va = verdict.annotate(batch_result, nearby_counts)
    vb = verdict.annotate(single_result, nearby_counts)
    key = lambda v: [(s["taxon_id"], s["verdict"], s["verdict_text"], s["best_day"]) for s in v["species"]]
    return {"max_abs_diff": max_diff, "rows": int(a.shape[0]) if same_shape else None,
            "other_fields_equal": _skeleton(batch_result) == _skeleton(single_result),
            "verdicts_equal": key(va) == key(vb), "headline_equal": va["headline"] == vb["headline"],
            "headline": va["headline"]}


def equivalence_check(cells: Sequence[dict], results: Dict[str, dict], check_ids: Sequence[str],
                      single_fn: Callable[[dict], dict], nearby: Dict[str, Optional[dict]],
                      tolerance: float = TOLERANCE, log=print) -> dict:
    by_id = {c["id"]: c for c in cells}
    rows = []
    for cid in check_ids:
        cell = by_id[cid]
        t = time.time()
        single = single_fn(cell)
        c = compare(results[cid], single, nearby.get(cid))
        c.update(cell=cid, lat=cell["lat"], lon=cell["lon"], single_seconds=round(time.time() - t, 1),
                 single_model=single["model"].get("name"))
        c["passed"] = bool(c["max_abs_diff"] <= tolerance and c["other_fields_equal"]
                           and c["verdicts_equal"] and c["headline_equal"])
        log("check %s: max |batch - single| = %.3g, verdicts %s, headline %s -> %s" % (
            cid, c["max_abs_diff"], "equal" if c["verdicts_equal"] else "DIFFER",
            "equal" if c["headline_equal"] else "DIFFERS", "ok" if c["passed"] else "FAIL"))
        rows.append(c)
    return {"tolerance": tolerance, "passed": bool(rows) and all(r["passed"] for r in rows),
            "max_abs_diff": max((r["max_abs_diff"] for r in rows), default=None), "cells": rows}


# ----------------------------------------------------------------------------- site files

def snapshot(cell: dict, annotated: dict, html: str, today: dt.date) -> dict:
    """The cell's file: the shape of a done /api/forecast/status reply, plus the cell it covers."""
    return {"id": cell["id"], "status": "done", "mode": "static", "expect": None,
            "place": {"lat": cell["lat"], "lon": cell["lon"], "name": annotated["place"]["name"]},
            "elapsed": 0.0, "result": annotated, "card_html": html, "seconds": None,
            "cell": dict({k: cell[k] for k in ("id", "lat", "lon", "lat_min", "lat_max", "lon_min", "lon_max")},
                         deg=config.GRID_DEG, note=cell_note(cell, today))}


def cell_file(cell: dict) -> str:
    return "cells/%s.json" % cell["id"]


def index_json(cells: Sequence[dict], today: dt.date, check: dict, model_info: dict, synthetic: bool) -> dict:
    return {
        "mode": "static", "synthetic": bool(synthetic),
        "today": today.isoformat(), "today_label": verdict.fmt_day(today), "timezone": config.TIMEZONE,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "update": UPDATE_NOTE, "grid_deg": config.GRID_DEG, "cell_rule": CELL_RULE,
        "region": dict(config.REGION), "region_note": config.REGION_NOTE,
        "model": model_info,
        "check": {k: check[k] for k in ("passed", "tolerance", "max_abs_diff")} | {
            "cells": [r["cell"] for r in check["cells"]]},
        "cells": [dict({k: c[k] for k in ("id", "lat", "lon", "lat_min", "lat_max", "lon_min", "lon_max")},
                       file=cell_file(c)) for c in cells],
    }


def static_index_html(source: str) -> str:
    """static/index.html with the static-mode marker added after the charset meta."""
    anchor = '<meta charset="utf-8">'
    if source.count(anchor) != 1 or "peakweek-static" in source:
        raise ValueError("unexpected static/index.html")
    return source.replace(anchor, anchor + "\n" + STATIC_META)


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))


def write_site(out_dir: Path, cells: Sequence[dict], snapshots: Dict[str, dict], index: dict, check: dict,
               static_dir: Path = STATIC_DIR) -> None:
    """Write the whole site to a scratch directory, then swap it into place (never half a site)."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "data" / "cells").mkdir(parents=True)
    for name in STATIC_ASSETS:
        shutil.copyfile(static_dir / name, tmp / name)
    (tmp / "index.html").write_text(static_index_html((static_dir / "index.html").read_text()))
    (tmp / ".nojekyll").write_text("")
    for c in cells:
        _dump(tmp / "data" / cell_file(c), snapshots[c["id"]])
    _dump(tmp / "data" / "index.json", index)
    _dump(tmp / "data" / "check.json", check)
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)


def render_cells(cells: Sequence[dict], results: Dict[str, dict], nearby: Dict[str, Optional[dict]],
                 today: dt.date, synthetic: bool = False) -> Dict[str, dict]:
    """annotate + render_html (both unchanged) for every cell -> {cell_id: snapshot}."""
    out = {}
    for c in cells:
        raw = results[c["id"]]
        if synthetic:
            raw = dict(raw, synthetic=True)
        annotated = verdict.annotate(raw, nearby.get(c["id"]))
        out[c["id"]] = snapshot(c, annotated, card.render_html(annotated), today)
    return out
