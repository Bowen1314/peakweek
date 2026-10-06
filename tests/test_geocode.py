import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from peakweek import geocode

SAVED = Path(__file__).resolve().parent / "data" / "geocode_new_brunswick.json"


class FakeOpener:
    """Stands in for urllib.request.urlopen; never touches the network."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.urls = []

    def __call__(self, req, timeout=None):
        self.urls.append(req.full_url)
        return io.BytesIO(json.dumps(self.responses.pop(0)).encode("utf-8"))


class LatLonTests(unittest.TestCase):
    def test_parses_pairs(self):
        self.assertEqual(geocode.parse_latlon("40.49, -74.45"), (40.49, -74.45))
        self.assertEqual(geocode.parse_latlon("40.49,-74.45"), (40.49, -74.45))
        self.assertEqual(geocode.parse_latlon("  40.49 -74.45 "), (40.49, -74.45))
        self.assertEqual(geocode.parse_latlon("+41, -73.5"), (41.0, -73.5))

    def test_not_a_pair(self):
        for text in ("New Brunswick, NJ", "40.49", "", "40.49, -74.45, 3", "lat 40 lon -74"):
            self.assertIsNone(geocode.parse_latlon(text), text)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            geocode.parse_latlon("100, -74")
        with self.assertRaises(ValueError):
            geocode.parse_latlon("40, -200")


class ResponseTests(unittest.TestCase):
    def test_saved_response(self):
        with open(SAVED, encoding="utf-8") as f:
            places = geocode.parse_response(json.load(f))
        self.assertGreaterEqual(len(places), 1)
        top = places[0]
        self.assertEqual(top["short"], "New Brunswick, NJ")
        self.assertEqual(top["label"], "New Brunswick, New Jersey, United States")
        self.assertAlmostEqual(top["lat"], 40.48622)
        self.assertAlmostEqual(top["lon"], -74.45182)

    def test_skips_bad_rows_and_handles_empty(self):
        data = {"results": [{"name": "X"}, {"name": "", "latitude": 1, "longitude": 2},
                            {"name": "Toronto", "latitude": 43.7, "longitude": -79.4,
                             "admin1": "Ontario", "country": "Canada", "country_code": "CA"}]}
        places = geocode.parse_response(data)
        self.assertEqual([p["short"] for p in places], ["Toronto, Ontario"])
        self.assertEqual(geocode.parse_response({"generationtime_ms": 0.5}), [])


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache = Path(self.tmp) / "geocode.json"
        with open(SAVED, encoding="utf-8") as f:
            self.saved = json.load(f)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_uses_cache_on_second_call(self):
        opener = FakeOpener(self.saved)
        first = geocode.search("New Brunswick, NJ", cache_path=self.cache, opener=opener)
        second = geocode.search("new  brunswick, nj", cache_path=self.cache, opener=opener)
        self.assertEqual(first, second)
        self.assertEqual(len(opener.urls), 1)
        self.assertIn("name=New+Brunswick%2C+NJ", opener.urls[0])
        self.assertTrue(self.cache.exists())

    def test_no_cache(self):
        opener = FakeOpener(self.saved, self.saved)
        geocode.search("New Brunswick, NJ", cache_path=None, opener=opener)
        geocode.search("New Brunswick, NJ", cache_path=None, opener=opener)
        self.assertEqual(len(opener.urls), 2)
        self.assertFalse(self.cache.exists())

    def test_coordinates_skip_the_network(self):
        opener = FakeOpener()
        places = geocode.search("40.49, -74.45", cache_path=None, opener=opener)
        self.assertEqual(places[0]["lat"], 40.49)
        self.assertEqual(opener.urls, [])

    def test_falls_back_to_town_name(self):
        opener = FakeOpener({}, self.saved)
        places = geocode.search("New Brunswick, Garden State", cache_path=None, opener=opener)
        self.assertEqual(places[0]["short"], "New Brunswick, NJ")
        self.assertIn("name=New+Brunswick&", opener.urls[1])


if __name__ == "__main__":
    unittest.main()
