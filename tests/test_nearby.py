import io
import json
import shutil
import socket
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
from pathlib import Path

from peakweek import nearby

SAVED = Path(__file__).resolve().parent / "data" / "inat_species_counts_new_brunswick.json"
IDS = [48098, 52543, 49658, 49005, 49202, 54802, 54795, 54763]


def load_saved():
    with open(SAVED, encoding="utf-8") as f:
        return json.load(f)


class FakeOpener:
    """Stands in for urllib.request.urlopen; never touches the network."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append((req, timeout))
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        if isinstance(r, bytes):
            return io.BytesIO(r)
        return io.BytesIO(json.dumps(r).encode("utf-8"))


class ParseTests(unittest.TestCase):
    def test_saved_response(self):
        # saved from the live API for New Brunswick, NJ (40.49, -74.45), 2026-10-06
        got = nearby.parse_response(load_saved(), IDS)
        self.assertEqual(got, {48098: 2721, 52543: 770, 49658: 3770, 49005: 845,
                               49202: 3883, 54802: 541, 54795: 2566, 54763: 2224})

    def test_missing_taxa_are_zero(self):
        data = {"total_results": 1, "results": [{"count": 3, "taxon": {"id": 48098, "ancestor_ids": [48098]}}]}
        got = nearby.parse_response(data, IDS)
        self.assertEqual(got[48098], 3)
        self.assertEqual(sorted(got), sorted(IDS))
        self.assertEqual(sum(got.values()), 3)
        self.assertEqual(nearby.parse_response({"results": []}, IDS), {t: 0 for t in IDS})

    def test_variety_counts_toward_its_species(self):
        data = {"results": [
            {"count": 10, "taxon": {"id": 52543, "ancestor_ids": [48460, 52543]}},
            {"count": 4, "taxon": {"id": 999001, "rank": "variety", "ancestor_ids": [48460, 47727, 52543, 999001]}},
            {"count": 7, "taxon": {"id": 123, "ancestor_ids": [48460, 123]}},  # not asked for: ignored
        ]}
        got = nearby.parse_response(data, IDS)
        self.assertEqual(got[52543], 14)
        self.assertNotIn(123, got)

    def test_bad_shapes_raise(self):
        for bad in (None, [], {"error": "x"}, {"results": "nope"}):
            with self.assertRaises(ValueError):
                nearby.parse_response(bad, IDS)


class CacheKeyTests(unittest.TestCase):
    def test_rounds_to_two_decimals(self):
        self.assertEqual(nearby.cache_key(40.4862, -74.4518), "40.49_-74.45")
        self.assertEqual(nearby.cache_key(44.4759, -73.2121), "44.48_-73.21")
        self.assertEqual(nearby.cache_key(40.4406, -79.9959), "40.44_-80.00")

    def test_nearby_points_share_a_key_and_no_negative_zero(self):
        self.assertEqual(nearby.cache_key(40.4851, -74.4549), nearby.cache_key(40.4862, -74.4518))
        self.assertEqual(nearby.cache_key(-0.001, 0.004), "0.00_0.00")

    def test_query_url(self):
        url = nearby.query_url(40.4862, -74.4518, IDS)
        parts = urllib.parse.urlsplit(url)
        q = dict(urllib.parse.parse_qsl(parts.query))
        self.assertEqual(parts.netloc + parts.path, "api.inaturalist.org/v1/observations/species_counts")
        self.assertEqual(q, {"lat": "40.49", "lng": "-74.45", "radius": "50",
                             "taxon_id": ",".join(str(t) for t in IDS),
                             "quality_grade": "research", "verifiable": "true"})


class CountsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.dir = self.tmp / "nearby"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fetch_then_cache_hit(self):
        op = FakeOpener(load_saved())
        first = nearby.counts(40.4862, -74.4518, IDS, cache_dir=self.dir, opener=op)
        self.assertEqual(first[49202], 3883)
        req, timeout = op.requests[0]
        self.assertEqual(req.get_header("User-agent"), "peakweek/0.1 (hackathon project; contact via GitHub)")
        self.assertEqual(timeout, 10.0)
        self.assertTrue((self.dir / "40.49_-74.45.json").exists())
        # same 0.01-degree cell: answered from disk, the (failing) opener is never called
        broken = FakeOpener(urllib.error.URLError("offline"))
        self.assertEqual(nearby.counts(40.4851, -74.4549, IDS, cache_dir=self.dir, opener=broken), first)
        self.assertEqual(broken.requests, [])

    def test_failures_return_none_and_are_not_cached(self):
        for failure in (urllib.error.URLError("offline"), socket.timeout("timed out"),
                        urllib.error.HTTPError("u", 503, "busy", {}, None), b"<html>not json</html>",
                        {"error": "rate limited"}):
            op = FakeOpener(failure)
            self.assertIsNone(nearby.counts(40.49, -74.45, IDS, cache_dir=self.dir, opener=op), failure)
        self.assertFalse((self.dir / "40.49_-74.45.json").exists())

    def test_stale_or_other_species_cache_is_refetched(self):
        path = self.dir / "40.49_-74.45.json"
        self.dir.mkdir(parents=True)
        with open(path, "w") as f:
            json.dump({"t": time.time() - nearby.CACHE_TTL - 1, "counts": {str(t): 1 for t in IDS}}, f)
        op = FakeOpener(load_saved())
        self.assertEqual(nearby.counts(40.49, -74.45, IDS, cache_dir=self.dir, opener=op)[48098], 2721)
        with open(path, "w") as f:
            json.dump({"t": time.time(), "counts": {"48098": 1}}, f)  # cached for fewer species
        op = FakeOpener(load_saved())
        self.assertEqual(nearby.counts(40.49, -74.45, IDS, cache_dir=self.dir, opener=op)[52543], 770)
        self.assertEqual(len(op.requests), 1)

    def test_corrupt_cache_is_ignored(self):
        self.dir.mkdir(parents=True)
        (self.dir / "40.49_-74.45.json").write_text("{nope")
        op = FakeOpener(load_saved())
        self.assertEqual(nearby.counts(40.49, -74.45, IDS, cache_dir=self.dir, opener=op)[54802], 541)

    def test_no_cache_dir(self):
        op = FakeOpener(load_saved(), load_saved())
        nearby.counts(40.49, -74.45, IDS, cache_dir=None, opener=op)
        nearby.counts(40.49, -74.45, IDS, cache_dir=None, opener=op)
        self.assertEqual(len(op.requests), 2)
        self.assertFalse(self.dir.exists())

    def test_default_ids_are_the_core_species(self):
        self.assertEqual(sorted(nearby.default_taxon_ids()), sorted(IDS))
        self.assertEqual(sorted(nearby._FALLBACK_IDS), sorted(IDS))


if __name__ == "__main__":
    unittest.main()
