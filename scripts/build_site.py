#!/usr/bin/env python3
"""Build the daily static site (GitHub Pages) from one batched TabPFN prediction. See peakweek/site.py.

    python3 scripts/build_site.py --out _site --report build-report     # the daily job (needs TabPFN)
    python3 scripts/build_site.py --fake --out /tmp/site                 # no model, no network: page check

Steps: fetch weather for every published cell (multi-location Open-Meteo requests) and their elevations,
then go offline; predict every cell's rows in one batch; re-run a few cells through forecast() and stop
unless the probabilities match; annotate, render and write the site. "today" is the New York date.
Any failure exits non-zero before the site directory is written, so the workflow publishes nothing and
GitHub Pages keeps serving the previous day's site.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from peakweek import config, forecast, site, weather  # noqa: E402


def _maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="_site", help="site directory to write (replaced as a whole)")
    p.add_argument("--report", default=None, help="also write check.json and build.json here (always, even on failure)")
    p.add_argument("--today", help="YYYY-MM-DD (default: today's date in America/New_York)")
    p.add_argument("--chunk", type=int, default=site.DEFAULT_CHUNK, help="query rows per predict_proba call")
    p.add_argument("--check", default=",".join(site.DEFAULT_CHECK), help="cell ids to re-run through forecast()")
    p.add_argument("--tolerance", type=float, default=site.TOLERANCE)
    p.add_argument("--limit", type=int, default=0, help="only the first N cells (plus the check cells); for tests")
    p.add_argument("--fake", action="store_true",
                   help="synthetic weather and a fake classifier, no network and no TabPFN; marks the site synthetic")
    args = p.parse_args(argv)

    t0 = time.time()
    timings = {}
    report_dir = Path(args.report) if args.report else None
    build = {"started_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "fake": args.fake}

    def save_report(check=None, ok=False, error=None):
        build.update(ok=ok, error=error, timings=timings, maxrss_mb=_maxrss_mb(),
                     total_seconds=round(time.time() - t0, 1))
        if report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "build.json").write_text(json.dumps(build, indent=1))
            if check is not None:
                (report_dir / "check.json").write_text(json.dumps(check, indent=1))

    try:
        today = dt.date.fromisoformat(args.today) if args.today else weather.local_today()
        build["today"] = today.isoformat()
        cells = site.published_cells()
        check_ids = [c for c in args.check.split(",") if c]
        unknown = set(check_ids) - {c["id"] for c in cells}
        if unknown:
            raise SystemExit("unknown check cells: %s" % sorted(unknown))
        if args.limit:
            cells = [c for i, c in enumerate(cells) if i < args.limit or c["id"] in check_ids]
        build.update(cells=len(cells), chunk=args.chunk, check_cells=check_ids)
        print("peakweek site: %d cells, today %s (America/New_York)" % (len(cells), today), flush=True)

        nearby = site.load_nearby()
        missing = [c["id"] for c in cells if nearby.get(c["id"]) is None]
        if missing and not args.fake:
            raise SystemExit("data/nearby_cells.json has no counts for %d cells (run scripts/build_nearby.py): %s"
                             % (len(missing), missing[:5]))

        points = [(c["lat"], c["lon"]) for c in cells]
        if args.fake:
            from tests.helpers import FakeClassifier, synthetic_forecast_weather
            order = {c["id"]: i for i, c in enumerate(site.published_cells())}

            def inject(cell):
                return {"weather_daily": synthetic_forecast_weather(today, seed=order[cell["id"]]), "elevation_m": 100.0}

            clf_factory, model_name = FakeClassifier, "synthetic test classifier (not TabPFN)"
        else:
            t = time.time()
            build["weather"] = weather.prefetch_app_daily(points, today)
            elev = weather.elevations(points)
            if any(e is None for e in elev):
                raise RuntimeError("elevation missing for %d cells" % sum(e is None for e in elev))
            timings["weather_seconds"] = round(time.time() - t, 1)
            os.environ["PEAKWEEK_OFFLINE"] = "1"  # from here on every weather/elevation read must hit the cache
            inject, clf_factory, model_name = None, None, forecast.TABPFN_NAME

        t = time.time()
        results, stats = site.batch_forecast(cells, today, clf_factory=clf_factory, model_name=model_name,
                                             inject=inject, chunk=args.chunk)
        timings["batch_seconds"] = round(time.time() - t, 1)
        timings["predict_seconds"] = stats["predict_seconds"]
        build["batch"] = stats
        build["maxrss_mb_after_batch"] = _maxrss_mb()

        def single(cell):
            if args.fake:
                return forecast.forecast(cell["lat"], cell["lon"], site.cell_name(cell), today,
                                         classifier=FakeClassifier(), **inject(cell))
            return forecast.forecast(cell["lat"], cell["lon"], site.cell_name(cell), today)

        t = time.time()
        check = site.equivalence_check(cells, results, check_ids, single, nearby, args.tolerance)
        check.update(today=today.isoformat(), chunk=args.chunk, fake=args.fake,
                     batch_model=model_name, single_path="peakweek.forecast.forecast(lat, lon, name, today)")
        timings["check_seconds"] = round(time.time() - t, 1)
        if not check["passed"]:
            save_report(check, ok=False, error="batch and single-point forecasts differ")
            print("FAILED: batch and single-point forecasts differ; nothing written", file=sys.stderr)
            return 2

        t = time.time()
        snaps = site.render_cells(cells, results, nearby, today, synthetic=args.fake)
        model_info = {"name": model_name, "config": dict(forecast.LOCKED_CONFIG), **{
            k: stats[k] for k in ("cells", "rows", "chunk", "predict_seconds", "context_rows")}}
        index = site.index_json(cells, today, check, model_info, synthetic=args.fake)
        site.write_site(Path(args.out), cells, snaps, index, check)
        timings["render_seconds"] = round(time.time() - t, 1)
        save_report(check, ok=True)
        print("wrote %s: %d cells; predict %.1f s; total %.1f s; max |batch - single| %.3g" % (
            args.out, len(cells), stats["predict_seconds"], time.time() - t0, check["max_abs_diff"]), flush=True)
        return 0
    except BaseException as e:  # record why, then fail the job
        if isinstance(e, SystemExit) and e.code in (0, None):
            raise
        save_report(None, ok=False, error="%s: %s" % (type(e).__name__, e))
        raise


if __name__ == "__main__":
    sys.exit(main())
