"""Static mode of the page (static/app.js, the GitHub Pages site): the pure helpers run under node and
must agree with the Python they mirror. Skipped when node is not installed."""

import json
import random
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from peakweek import config, geocode, site, weather

ROOT = Path(__file__).resolve().parent.parent
APP_JS = ROOT / "static" / "app.js"
GEOCODE_SAVED = ROOT / "tests" / "data" / "geocode_new_brunswick.json"
NODE = shutil.which("node")

DRIVER = r"""
const fs = require("fs"), vm = require("vm");
const src = fs.readFileSync(process.argv[2], "utf8");
const block = src.slice(src.indexOf("// BEGIN static helpers"), src.indexOf("// END static helpers"));
const S = vm.runInNewContext(block + "\nPeakweekStatic;", { URLSearchParams });
const input = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const latlon = input.latlon.map((t) => { try { return { ok: S.parseLatLon(t) }; } catch (e) { return { error: e.message }; } });
const cells = input.points.map(([a, b]) => { const c = S.findCell(input.cells, a, b); return c ? c.id : null; });
const region = input.points.map(([a, b]) => S.inRegion(input.region, a, b));
const places = S.parsePlaces(input.geocode);
const coords = input.coords.map(([a, b]) => S.latlonPlace(a, b));
process.stdout.write(JSON.stringify({ latlon, cells, region, places, coords, states: S.US_STATES,
  url: S.geocodeURL("New Brunswick, NJ") }));
"""

LATLON_CASES = ["40.49, -74.45", "40.49 -74.45", " +40.4862,-74.4518 ", ".5, -.5", "40.", "Ithaca, NY",
                "91, 0", "0, -181", "40.49,", "1e3, 2", "-90, 180", "40.49 , -74.45"]


@unittest.skipUnless(NODE, "node is not installed")
class StaticHelpersMatchPython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(1)
        cls.cells = site.published_cells()
        cls.points = [[rng.uniform(37.0, 49.0), rng.uniform(-82.0, -65.0)] for _ in range(3000)]
        cls.points += [[40.0, -74.0], [41.0, -75.0], [47.5, -66.9], [38.5, -80.5], [47.5000001, -70.0],
                       [38.4999999, -75.0], [44.6, -67.4], [45.6, -70.4], [33.749, -84.388]]
        saved = json.loads(GEOCODE_SAVED.read_text())
        extra = {"results": [
            {"name": "Montreal", "latitude": 45.5, "longitude": -73.57, "admin1": "Quebec", "country": "Canada",
             "country_code": "CA"},
            {"name": "  ", "latitude": 1, "longitude": 2},
            {"name": "No coords"},
            {"name": "Odd", "latitude": "x", "longitude": 2},
            {"name": "Nowhere", "latitude": 10, "longitude": 20, "country": "Atlantis"},
            {"name": "Proto", "latitude": 10, "longitude": 20, "admin1": "constructor", "country_code": "US"},
        ]}
        cls.geocode_input = {"results": saved["results"] + extra["results"]}
        cls.coords = [[40.4862, -74.4518], [-0.5, 179.99995], [12.0, -3.25]]
        payload = {"latlon": LATLON_CASES, "points": cls.points, "region": config.REGION,
                   "cells": [{k: c[k] for k in ("id", "lat_min", "lat_max", "lon_min", "lon_max")} for c in cls.cells],
                   "geocode": cls.geocode_input, "coords": cls.coords}
        with tempfile.TemporaryDirectory() as tmp:
            drv, inp = Path(tmp) / "driver.js", Path(tmp) / "input.json"
            drv.write_text(DRIVER)
            inp.write_text(json.dumps(payload))
            out = subprocess.run([NODE, str(drv), str(APP_JS), str(inp)], capture_output=True, text=True, timeout=60)
        if out.returncode:
            raise AssertionError(out.stderr)
        cls.js = json.loads(out.stdout)

    def test_latlon_parsing(self):
        for text, got in zip(LATLON_CASES, self.js["latlon"]):
            try:
                want = geocode.parse_latlon(text)
            except ValueError as e:
                self.assertEqual(got, {"error": str(e)}, text)
                continue
            self.assertEqual(got, {"ok": list(want) if want else None}, text)

    def test_cell_lookup_matches_cell_id(self):
        ids = {c["id"] for c in self.cells}
        for (lat, lon), got in zip(self.points, self.js["cells"]):
            cid = weather.cell_id(lat, lon)
            self.assertEqual(got, cid if cid in ids else None, (lat, lon))

    def test_region_check(self):
        for (lat, lon), got in zip(self.points, self.js["region"]):
            self.assertEqual(got, config.in_region(lat, lon), (lat, lon))

    def test_place_parsing(self):
        self.assertEqual(self.js["places"], geocode.parse_response(self.geocode_input))
        self.assertEqual(self.js["states"], geocode.US_STATES)
        self.assertEqual(self.js["coords"], [geocode.latlon_place(a, b) for a, b in self.coords])

    def test_geocode_url_matches_server(self):
        self.assertEqual(self.js["url"], geocode.API_URL + "?name=New+Brunswick%2C+NJ&count=5&language=en&format=json")


class StaticPageTests(unittest.TestCase):
    def test_static_mode_uses_relative_paths_only(self):
        src = APP_JS.read_text()
        block = src[src.index("// ---- static mode"):src.index("const fmtInt")]
        for url in re.findall(r'getJSON\(\s*"([^"]+)"', block):
            self.assertTrue(url.startswith("./"), url)  # GitHub Pages serves the site under /peakweek/
        self.assertIn('getJSON("./data/" + cell.file)', block)
        html = site.static_index_html((ROOT / "static" / "index.html").read_text())
        for val in re.findall(r'(?:src|href)="([^"]*)"', html):
            self.assertFalse(val.startswith("/"), val)

    def test_live_mode_is_unchanged_without_the_marker(self):
        html = (ROOT / "static" / "index.html").read_text()
        self.assertNotIn("peakweek-static", html)  # server.py serves this file: live mode
        src = APP_JS.read_text()
        self.assertIn("return STATIC_INDEX ? staticGet(url) : getJSON(url);", src)


if __name__ == "__main__":
    unittest.main()
