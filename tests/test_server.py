import copy
import json
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import server

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "forecast_fixture.json"  # synthetic, New Brunswick (40.49, -74.45)
REAL_FIXTURE = ROOT / "examples" / "forecast_new_brunswick.json"  # saved real forecast (40.4862, -74.4518)
KNOWN_IDS = (48098, 52543, 49658, 49005, 49202, 54802, 54795, 54763)


def load_fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


class FakeForecast:
    """Injected in place of peakweek.forecast.forecast; records calls and overlap."""

    def __init__(self, delay=0.0, fail=None):
        self.delay = delay
        self.fail = fail
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, lat, lon, name):
        with self.lock:
            self.calls.append((lat, lon, name))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            if self.fail:
                raise RuntimeError(self.fail)
            r = load_fixture()
            r.pop("synthetic", None)
            r["place"] = {"name": name, "lat": lat, "lon": lon}
            r["model"]["context_rows"] = 777
            return r
        finally:
            with self.lock:
                self.active -= 1


class FakeNearby:
    """Injected in place of peakweek.nearby.counts; records the place asked about and the thread."""

    def __init__(self, counts=None, fail=None, delay=0.0):
        self.counts = counts if counts is not None else {t: 300 for t in KNOWN_IDS}
        self.fail = fail
        self.delay = delay
        self.calls = []

    def __call__(self, lat, lon):
        self.calls.append((lat, lon, threading.current_thread().name))
        if self.delay:
            time.sleep(self.delay)
        if self.fail == "none":
            return None
        if self.fail:
            raise RuntimeError(self.fail)
        return dict(self.counts)


class ServerCase(unittest.TestCase):
    def start(self, **kw):
        kw.setdefault("seed_example", None)
        kw.setdefault("nearby_fn", FakeNearby())  # tests never call iNaturalist
        self.srv = server.make_server("127.0.0.1", 0, quiet=True, **kw)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:%d" % self.srv.server_address[1]

    def tearDown(self):
        srv = getattr(self, "srv", None)
        if srv:
            srv.shutdown()
            srv.server_close()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def get_json(self, path):
        code, ctype, body = self.get(path)
        self.assertIn("application/json", ctype)
        return code, json.loads(body)

    def wait_done(self, job, timeout=10):
        t0 = time.time()
        while job["status"] in ("queued", "running"):
            self.assertLess(time.time() - t0, timeout, "forecast did not finish")
            time.sleep(0.05)
            code, job = self.get_json("/api/forecast/status?id=" + job["id"])
            self.assertEqual(code, 200)
        return job


class FixtureModeTests(ServerCase):
    def test_info_page_and_forecast(self):
        self.start(fixture=FIXTURE)
        code, info = self.get_json("/api/info")
        self.assertEqual(info, {"mode": "fixture", "fixture_place": "New Brunswick, NJ", "synthetic": True})
        code, ctype, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"peakweek", body)
        for path in ("/app.js", "/style.css"):
            self.assertEqual(self.get(path)[0], 200, path)

        code, job = self.get_json("/api/forecast?lat=40.49&lon=-74.45&name=New%20Brunswick%2C%20NJ")
        self.assertEqual(code, 200)
        self.assertEqual(job["mode"], "fixture")
        self.assertEqual(job["expect"], {"past_days": 92, "context_rows": 1000, "seconds": 21.3})
        job = self.wait_done(job)
        self.assertEqual(job["status"], "done")
        r = job["result"]
        self.assertTrue(r["synthetic"])
        self.assertTrue(r["headline"].startswith("This weekend, go see"))
        self.assertIn("banner-synthetic", job["card_html"])
        for sp in r["species"]:
            self.assertAlmostEqual(sum(sp["weekend"].values()), 1.0, delta=1e-6)

        code, again = self.get_json("/api/forecast?lat=40.491&lon=-74.449")
        self.assertEqual(again["status"], "done")
        self.assertTrue(again.get("cached"))


class LiveModeTests(ServerCase):
    def test_uses_injected_forecast_and_caches(self):
        fake = FakeForecast()
        self.start(forecast_fn=fake)
        code, info = self.get_json("/api/info")
        self.assertEqual(info["mode"], "live")
        code, job = self.get_json("/api/forecast?lat=42.44&lon=-76.5&name=Ithaca%2C%20NY")
        self.assertIsNone(job["expect"])  # nothing known yet: the page shows no numbers
        job = self.wait_done(job)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["place"]["name"], "Ithaca, NY")
        self.assertNotIn("synthetic", job["result"])
        self.assertEqual(fake.calls, [(42.44, -76.5, "Ithaca, NY")])

        code, again = self.get_json("/api/forecast?lat=42.4401&lon=-76.5001")
        self.assertEqual(again["status"], "done")
        self.assertEqual(len(fake.calls), 1)

        code, other = self.get_json("/api/forecast?lat=41.0&lon=-74.0")
        self.assertEqual(other["expect"]["context_rows"], 777)  # learned from the finished run
        self.wait_done(other)
        self.assertEqual(len(fake.calls), 2)

    def test_one_forecast_at_a_time(self):
        fake = FakeForecast(delay=0.3)
        self.start(forecast_fn=fake)
        _, a = self.get_json("/api/forecast?lat=40&lon=-74")
        time.sleep(0.05)
        _, b = self.get_json("/api/forecast?lat=41&lon=-74")
        self.assertEqual(b["status"], "queued")
        self.assertGreaterEqual(b["ahead"], 1)
        self.wait_done(a)
        self.wait_done(b)
        self.assertEqual(fake.max_active, 1)

    def test_error_is_reported_and_retried(self):
        fake = FakeForecast(fail="weather API down")
        self.start(forecast_fn=fake)
        _, job = self.get_json("/api/forecast?lat=40&lon=-74")
        job = self.wait_done(job)
        self.assertEqual(job["status"], "error")
        self.assertIn("weather API down", job["error"])
        _, retry = self.get_json("/api/forecast?lat=40&lon=-74")
        self.assertNotEqual(retry["id"], job["id"])
        self.wait_done(retry)
        self.assertEqual(len(fake.calls), 2)

    def test_bad_requests(self):
        self.start(forecast_fn=FakeForecast())
        self.assertEqual(self.get_json("/api/forecast?lat=100&lon=0")[0], 400)
        self.assertEqual(self.get_json("/api/forecast?lat=abc&lon=0")[0], 400)
        self.assertEqual(self.get_json("/api/forecast?lat=nan&lon=0")[0], 400)
        self.assertEqual(self.get_json("/api/forecast?lon=0")[0], 400)
        self.assertEqual(self.get_json("/api/forecast/status?id=nope")[0], 404)
        self.assertEqual(self.get_json("/api/nothing")[0], 404)
        self.assertEqual(self.get_json("/../server.py")[0], 404)
        self.assertEqual(self.get_json("/static/../server.py")[0], 404)


class GeocodeEndpointTests(ServerCase):
    def test_coordinates_without_network(self):
        def no_network(q):
            raise AssertionError("should not be called")
        self.start(forecast_fn=FakeForecast(), geocode_fn=no_network)
        code, data = self.get_json("/api/geocode?q=40.49%2C%20-74.45")
        self.assertEqual(code, 200)
        self.assertEqual((data["results"][0]["lat"], data["results"][0]["lon"]), (40.49, -74.45))
        self.assertEqual(self.get_json("/api/geocode?q=95%2C%200")[0], 400)
        self.assertEqual(self.get_json("/api/geocode?q=")[0], 400)

    def test_injected_search_and_failure(self):
        seen = []

        def fake_search(q):
            seen.append(q)
            if q == "boom":
                raise OSError("offline")
            return [{"name": "Ithaca", "short": "Ithaca, NY", "label": "Ithaca, New York, United States",
                     "lat": 42.44, "lon": -76.5, "country_code": "US"}]
        self.start(forecast_fn=FakeForecast(), geocode_fn=fake_search)
        code, data = self.get_json("/api/geocode?q=Ithaca%2C%20NY")
        self.assertEqual(code, 200)
        self.assertEqual(data["results"][0]["short"], "Ithaca, NY")
        code, data = self.get_json("/api/geocode?q=boom")
        self.assertEqual(code, 502)
        self.assertIn("coordinates", data["error"])
        self.assertEqual(seen, ["Ithaca, NY", "boom"])


class NearbyServerTests(ServerCase):
    def test_fixture_mode_checks_the_fixtures_place_off_the_worker(self):
        fake = FakeNearby(counts={t: (0 if t == 48098 else 300) for t in KNOWN_IDS})  # red maple rare here
        self.start(fixture=FIXTURE, nearby_fn=fake)
        _, job = self.get_json("/api/forecast?lat=44.4759&lon=-73.2121&name=Burlington%2C%20VT")
        job = self.wait_done(job)
        self.assertEqual(job["status"], "done")
        self.assertEqual(len(fake.calls), 1)
        lat, lon, thread = fake.calls[0]
        # the card shows the fixture's New Brunswick numbers, so the check is for New Brunswick too
        self.assertEqual((lat, lon), (40.49, -74.45))
        self.assertEqual(thread, "peakweek-nearby")
        self.assertNotEqual(thread, "peakweek-forecast")
        r = job["result"]
        self.assertTrue(r["nearby"]["available"])
        red = [s for s in r["species"] if s["taxon_id"] == 48098][0]
        self.assertTrue(red["rare_here"])
        self.assertEqual(red["nearby_observations"], 0)
        self.assertNotIn("red maple", r["headline"])
        self.assertIn(' rare">', job["card_html"])
        # the in-memory job cache also keeps the check: no second call for the same place and day
        _, again = self.get_json("/api/forecast?lat=44.476&lon=-73.212")
        self.assertTrue(again.get("cached"))
        self.assertEqual(len(fake.calls), 1)

    def test_live_mode_checks_the_requested_place(self):
        fake = FakeNearby()
        self.start(forecast_fn=FakeForecast(), nearby_fn=fake)
        _, job = self.get_json("/api/forecast?lat=42.44&lon=-76.5&name=Ithaca%2C%20NY")
        job = self.wait_done(job)
        self.assertEqual([c[:2] for c in fake.calls], [(42.44, -76.5)])
        self.assertNotIn("replay", job["result"])
        self.assertNotIn("banner-replay", job["card_html"])
        self.assertTrue(job["result"]["nearby"]["available"])
        self.assertFalse(any(s["rare_here"] for s in job["result"]["species"]))

    def test_failed_check_marks_unavailable(self):
        for fake in (FakeNearby(fail="none"), FakeNearby(fail="iNaturalist down")):
            self.start(fixture=FIXTURE, nearby_fn=fake)
            _, job = self.get_json("/api/forecast?lat=40.49&lon=-74.45")
            job = self.wait_done(job)
            self.assertEqual(job["status"], "done")
            self.assertFalse(job["result"]["nearby"]["available"])
            self.assertIn("Local check unavailable", job["card_html"])
            self.tearDown()
            self.srv = None

    def test_slow_check_is_not_waited_for_forever(self):
        fake = FakeNearby(delay=2.0)
        self.start(fixture=FIXTURE, nearby_fn=fake)
        self.srv.forecaster.nearby_wait = 0.2
        t0 = time.time()
        _, job = self.get_json("/api/forecast?lat=40.49&lon=-74.45")
        job = self.wait_done(job)
        self.assertLess(time.time() - t0, 1.5)
        self.assertFalse(job["result"]["nearby"]["available"])

    def test_no_check(self):
        self.start(fixture=FIXTURE, nearby_fn=None)
        _, job = self.get_json("/api/forecast?lat=40.49&lon=-74.45")
        job = self.wait_done(job)
        self.assertFalse(job["result"]["nearby"]["available"])


class ReplayNoticeTests(ServerCase):
    """Fixture mode with a saved real forecast says it is a replay, and of which place."""

    def card_for(self, query, fixture=REAL_FIXTURE):
        fake = FakeNearby()
        self.start(fixture=fixture, nearby_fn=fake)
        _, job = self.get_json("/api/forecast?" + query)
        job = self.wait_done(job)
        self.assertEqual(job["status"], "done")
        with open(fixture, encoding="utf-8") as f:
            place = json.load(f)["place"]
        self.assertEqual([c[:2] for c in fake.calls], [(place["lat"], place["lon"])])  # always the fixture's place
        return job

    def test_same_place(self):
        job = self.card_for("lat=40.4862&lon=-74.4518&name=New%20Brunswick%2C%20NJ")
        self.assertEqual(job["result"]["replay"], {"place": "New Brunswick, NJ", "made": "2026-10-06", "searched": None})
        self.assertIn('<p class="banner banner-replay" role="note">Replaying a saved forecast for New Brunswick, NJ '
                      '(made Oct 6). Run without --fixture for a live forecast.</p>', job["card_html"])
        self.assertNotIn("banner-synthetic", job["card_html"])

    def test_my_location_nearby_counts_as_the_same_place(self):
        job = self.card_for("lat=40.49&lon=-74.45&name=40.49%2C%20-74.45%20(your%20location)")
        self.assertIsNone(job["result"]["replay"]["searched"])

    def test_other_place(self):
        job = self.card_for("lat=44.4759&lon=-73.2121&name=Burlington%2C%20VT")
        self.assertEqual(job["result"]["replay"]["searched"], "Burlington, VT")
        self.assertIn("You searched Burlington, VT; this is the saved forecast for New Brunswick, NJ (made Oct 6). "
                      "Run without --fixture for a live forecast.", job["card_html"])
        self.assertEqual(job["result"]["place"]["name"], "New Brunswick, NJ")

    def test_other_place_without_name_and_escaping(self):
        job = self.card_for("lat=44.48&lon=-73.21")
        self.assertIn("You searched 44.48, -73.21; this is the saved forecast", job["card_html"])
        self.tearDown()
        job = self.card_for("lat=44.48&lon=-73.21&name=%3Cb%3EBurlington%3C%2Fb%3E")
        self.assertIn("You searched &lt;b&gt;Burlington&lt;/b&gt;;", job["card_html"])
        self.assertNotIn("<b>Burlington", job["card_html"])

    def test_synthetic_fixture_has_no_replay_notice(self):
        job = self.card_for("lat=44.4759&lon=-73.2121&name=Burlington%2C%20VT", fixture=FIXTURE)
        self.assertNotIn("replay", job["result"])
        self.assertNotIn("banner-replay", job["card_html"])
        self.assertIn("banner-synthetic", job["card_html"])


class ImportTests(unittest.TestCase):
    def test_server_import_does_not_load_the_model(self):
        code = ("import sys; sys.path.insert(0, %r); import server; "
                "print(any(m == 'tabpfn' or m.startswith('tabpfn.') or m == 'torch' for m in sys.modules), "
                "'peakweek.forecast' in sys.modules)" % str(ROOT))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ["False", "False"])

    def test_fixture_server_never_imports_forecast(self):
        code = (
            "import sys, time; sys.path.insert(0, %r); import server\n"
            "srv = server.make_server('127.0.0.1', 0, fixture=%r, quiet=True, nearby_fn=None)\n"
            "job = srv.forecaster.submit(40.49, -74.45, None)\n"
            "t0 = time.time()\n"
            "while srv.forecaster.status(job['id'])['status'] != 'done' and time.time() - t0 < 10: time.sleep(0.02)\n"
            "print(srv.forecaster.status(job['id'])['status'], 'peakweek.forecast' in sys.modules)\n"
            "srv.server_close()\n" % (str(ROOT), str(FIXTURE)))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT), timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ["done", "False"])

if __name__ == "__main__":
    unittest.main()
