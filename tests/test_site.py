"""The daily static site build (peakweek/site.py, scripts/build_site.py): no network, no TabPFN."""

import datetime as dt
import json
import os
import random
import tempfile
import threading
import time
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

import numpy as np

from peakweek import config, forecast, site, verdict, weather
from tests.helpers import FakeClassifier, synthetic_forecast_weather

ROOT = Path(__file__).resolve().parent.parent
TODAY = dt.date(2026, 10, 7)
CELLS = site.published_cells()
BY_ID = {c["id"]: c for c in CELLS}
SOME = [BY_ID[i] for i in ("38.50_-74.50", "40.50_-74.50", "44.50_-73.50", "47.50_-79.50")]


def inject(cell):
    seed = [c["id"] for c in CELLS].index(cell["id"])
    return {"weather_daily": synthetic_forecast_weather(TODAY, seed=seed), "elevation_m": 100.0 + seed}


def single(cell, today=TODAY):
    return forecast.forecast(cell["lat"], cell["lon"], site.cell_name(cell), today,
                             classifier=FakeClassifier(), **inject(cell))


class CellTests(unittest.TestCase):
    def test_published_cells_are_the_training_cells(self):
        self.assertEqual(len(CELLS), 103)
        self.assertEqual([c["id"] for c in CELLS], sorted(c["id"] for c in CELLS))
        for c in CELLS:
            self.assertAlmostEqual(c["lat_max"] - c["lat_min"], config.GRID_DEG)
            self.assertAlmostEqual(c["lon_max"] - c["lon_min"], config.GRID_DEG)
            self.assertTrue(config.in_region(c["lat"], c["lon"]), c["id"])
            self.assertIs(site.find_cell(CELLS, c["lat"], c["lon"]), c)

    def test_bounds_lookup_matches_cell_id(self):
        rng = random.Random(0)
        pts = [(rng.uniform(37.5, 48.5), rng.uniform(-81.5, -65.5)) for _ in range(5000)]
        pts += [(40.0, -74.0), (41.0, -75.0), (47.5, -66.9), (38.5, -80.5), (40.999999, -74.000001), (39.0, -77.0)]
        ids = {c["id"] for c in CELLS}
        for lat, lon in pts:
            found = site.find_cell(CELLS, lat, lon)
            cid = weather.cell_id(lat, lon)
            self.assertEqual(found["id"] if found else None, cid if cid in ids else None, (lat, lon))

    def test_cell_note_and_name(self):
        c = BY_ID["40.50_-74.50"]
        self.assertEqual(site.cell_note(c, TODAY),
                         "Forecast made Wed, Oct 7 for the 1° grid cell around 40.59, -74.45; updated daily.")
        self.assertEqual(site.cell_name(c), "1° grid cell around 40.59, -74.45")


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.stats = site.batch_forecast(SOME, TODAY, clf_factory=FakeClassifier, model_name="fake",
                                                     inject=inject, chunk=50, log=lambda *a: None)

    def test_one_fit_per_subsample_for_all_cells(self):
        self.assertEqual(self.stats["fits"], forecast.LOCKED_CONFIG["n_subsamples"])
        self.assertEqual(self.stats["rows"], 120 * len(SOME))

    def test_batch_equals_single_point(self):
        for c in SOME:
            cmp = site.compare(self.results[c["id"]], single(c))
            self.assertEqual(cmp["max_abs_diff"], 0.0, c["id"])
            self.assertTrue(cmp["other_fields_equal"], c["id"])
            self.assertTrue(cmp["verdicts_equal"] and cmp["headline_equal"], c["id"])

    def test_result_has_the_forecast_shape(self):
        r = self.results["40.50_-74.50"]
        s = single(BY_ID["40.50_-74.50"])
        self.assertEqual(set(r), set(s))
        self.assertEqual(set(r["model"]) - {"batch"}, set(s["model"]))
        self.assertEqual(r["today"], "2026-10-07")
        self.assertEqual(r["place"]["lat"], 40.59)
        self.assertIsNone(r["model"]["seconds"])

    def test_check_passes_and_catches_a_difference(self):
        ok = site.equivalence_check(SOME, self.results, ["40.50_-74.50", "47.50_-79.50"], single, {}, log=lambda *a: None)
        self.assertTrue(ok["passed"])
        self.assertEqual(ok["max_abs_diff"], 0.0)
        bad = json.loads(json.dumps(self.results))
        day = bad["40.50_-74.50"]["species"][3]["daily"][5]
        day["colored"] += 2e-5
        day["green"] -= 2e-5
        out = site.equivalence_check(SOME, bad, ["40.50_-74.50"], single, {}, log=lambda *a: None)
        self.assertFalse(out["passed"])
        self.assertAlmostEqual(out["max_abs_diff"], 2e-5, places=9)
        moved = json.loads(json.dumps(self.results))
        moved["44.50_-73.50"]["weather"]["archive_from"] = "2026-09-02"
        out = site.equivalence_check(SOME, moved, ["44.50_-73.50"], single, {}, log=lambda *a: None)
        self.assertFalse(out["passed"])
        self.assertFalse(out["cells"][0]["other_fields_equal"])

    def test_snapshot_has_the_server_status_shape(self):
        import server
        raw = self.results["40.50_-74.50"]
        f = server.Forecaster(lambda lat, lon, name: json.loads(json.dumps(raw)), "live", quiet=True, nearby_fn=None)
        job = f.submit(40.59, -74.45, "x")
        t0 = time.time()
        while job["status"] != "done":
            self.assertLess(time.time() - t0, 10)
            time.sleep(0.02)
            job = f.status(job["id"])
        snaps = site.render_cells(SOME, self.results, {}, TODAY)
        snap = snaps["40.50_-74.50"]
        self.assertEqual(set(snap) - {"cell"}, set(job))
        self.assertEqual(set(snap["result"]), set(job["result"]))
        self.assertEqual(snap["status"], "done")
        self.assertEqual(snap["mode"], "static")
        self.assertEqual(snap["result"]["headline"], job["result"]["headline"])
        self.assertIn('class="made"', snap["card_html"])
        self.assertEqual(snap["cell"]["note"], site.cell_note(BY_ID["40.50_-74.50"], TODAY))


class WriteSiteTests(unittest.TestCase):
    def test_fake_build_writes_a_relative_static_site(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            out, rep = Path(tmp) / "site", Path(tmp) / "report"
            with mock.patch("builtins.print"):
                code = mod.main(["--fake", "--limit", "2", "--today", "2026-10-07", "--out", str(out),
                                 "--report", str(rep)])
            self.assertEqual(code, 0)
            self.assertFalse((Path(tmp) / "site.partial").exists())
            html = (out / "index.html").read_text()
            self.assertEqual(html.count('name="peakweek-static"'), 1)
            for attr in ("src", "href"):
                for val in __import__("re").findall(r'%s="([^"]*)"' % attr, html):
                    self.assertFalse(val.startswith("/"), val)  # the site lives at /peakweek/, not at /
            self.assertEqual((out / "app.js").read_bytes(), (ROOT / "static" / "app.js").read_bytes())
            index = json.loads((out / "data" / "index.json").read_text())
            self.assertTrue(index["synthetic"])
            self.assertEqual(index["today"], "2026-10-07")
            self.assertEqual(index["today_label"], "Wed, Oct 7")
            self.assertEqual(index["region"], config.REGION)
            self.assertTrue(index["check"]["passed"])
            expected = {c["id"] for c in CELLS[:2]} | set(site.DEFAULT_CHECK)
            self.assertEqual({c["id"] for c in index["cells"]}, expected)
            for c in index["cells"]:
                self.assertFalse(c["file"].startswith("/"))
                snap = json.loads((out / "data" / c["file"]).read_text())
                self.assertEqual(snap["id"], c["id"])
                self.assertTrue(snap["result"]["synthetic"])
                self.assertIn("banner-synthetic", snap["card_html"])
                self.assertEqual(snap["result"]["days"][0], "2026-10-07")
            check = json.loads((rep / "check.json").read_text())
            self.assertTrue(check["passed"])
            self.assertEqual([r["cell"] for r in check["cells"]], list(site.DEFAULT_CHECK))
            build = json.loads((rep / "build.json").read_text())
            self.assertTrue(build["ok"])
            self.assertIn("predict_seconds", build["timings"])

    def test_static_index_html_refuses_surprises(self):
        src = (ROOT / "static" / "index.html").read_text()
        out = site.static_index_html(src)
        self.assertIn(site.STATIC_META, out)
        with self.assertRaises(ValueError):
            site.static_index_html(out)


def _fake_openmeteo(calls):
    """urlopen stand-in for multi-location forecast/archive requests: a list of per-location responses."""

    class Resp:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        url = req.full_url
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        lats = [float(x) for x in q["latitude"][0].split(",")]
        lons = [float(x) for x in q["longitude"][0].split(",")]
        calls.append((url.split("?")[0], len(lats)))
        if "archive" in url:
            start = dt.date.fromisoformat(q["start_date"][0])
            end = dt.date.fromisoformat(q["end_date"][0])
            n = (end - start).days + 1
        else:
            start = TODAY - dt.timedelta(days=int(q["past_days"][0]))
            n = int(q["past_days"][0]) + int(q["forecast_days"][0])
        out = []
        for lat, lon in zip(lats, lons):
            days = [(start + dt.timedelta(days=i)).isoformat() for i in range(n)]
            base = lat / 10 + (0.5 if "archive" in url else 0.0)
            out.append({"latitude": lat + 0.02, "longitude": lon - 0.03, "elevation": 10.0,
                        "daily": {"time": days, "temperature_2m_mean": [base + i * 0.01 for i in range(n)],
                                  "temperature_2m_min": [base - 5] * n, "precipitation_sum": [1.0] * n}})
        return Resp(json.dumps(out if len(out) > 1 else out[0]).encode())

    return opener


class PrefetchTests(unittest.TestCase):
    def test_multi_location_responses_fill_the_single_point_cache(self):
        pts = [(c["lat"], c["lon"]) for c in CELLS[:30]]
        calls = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(config, "CACHE_DIR", Path(tmp)), \
                mock.patch.object(weather.urllib.request, "urlopen", _fake_openmeteo(calls)), \
                mock.patch.dict(os.environ, {"PEAKWEEK_OFFLINE": "0"}):
            stats = weather.prefetch_app_daily(pts, TODAY, log=lambda *a: None)
            self.assertEqual(stats["points"], 30)
            self.assertEqual(sorted(n for _, n in calls), [5, 5, 25, 25])  # batches of <= 25 per API
            self.assertEqual(stats["archive_from"], "2026-09-01")
            self.assertEqual(stats["archive_to"], "2026-10-06")
            with mock.patch.dict(os.environ, {"PEAKWEEK_OFFLINE": "1"}):
                for lat, lon in pts[:: 7]:
                    daily, meta = weather.app_daily(lat, lon, TODAY)  # served from the cache, no network
                    self.assertEqual(meta["archive_from"], "2026-09-01")
                    row = daily[daily["date"] == "2026-09-10"].iloc[0]
                    self.assertAlmostEqual(row["tmean"], lat / 10 + 0.5 + 9 * 0.01)  # archive wins for the past
                    future = daily[daily["date"] == "2026-10-10"].iloc[0]
                    self.assertAlmostEqual(future["tmean"], lat / 10 + (92 + 3) * 0.01)
            n = len(calls)
            weather.prefetch_app_daily(pts, TODAY, log=lambda *a: None)
            self.assertEqual(len(calls), n)  # everything cached: no new request

    def test_split_rejects_short_or_misplaced_answers(self):
        pts = [(40.59, -74.45), (44.48, -72.59)]
        good = [{"latitude": 40.58, "longitude": -74.44, "daily": {}}, {"latitude": 44.47, "longitude": -72.62, "daily": {}}]
        self.assertEqual(len(weather._split_many(good, pts)), 2)
        with self.assertRaises(RuntimeError):
            weather._split_many(good[:1], pts)
        with self.assertRaises(RuntimeError):
            weather._split_many(list(reversed(good)), pts)


class NearbyFileTests(unittest.TestCase):
    def test_nearby_file_covers_every_published_cell(self):
        path = site.NEARBY_FILE
        if not path.exists():
            self.skipTest("data/nearby_cells.json not built yet")
        data = json.loads(path.read_text())
        self.assertEqual(data["radius_km"], verdict.NEARBY_RADIUS_KM)
        nearby = site.load_nearby()
        for c in CELLS:
            counts = nearby.get(c["id"])
            self.assertIsNotNone(counts, c["id"])
            self.assertEqual(sorted(int(k) for k in counts), sorted(s["taxon_id"] for s in forecast.SPECIES))
            self.assertEqual((data["cells"][c["id"]]["lat"], data["cells"][c["id"]]["lon"]), (c["lat"], c["lon"]))


if __name__ == "__main__":
    unittest.main()
