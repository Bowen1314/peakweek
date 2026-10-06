import contextlib
import io
import json
import unittest
from pathlib import Path

from peakweek import card, cli

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "forecast_fixture.json"
KNOWN_IDS = (48098, 52543, 49658, 49005, 49202, 54802, 54795, 54763)


def run(argv, nearby_fn):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv, nearby_fn=nearby_fn)
    return code, out.getvalue(), err.getvalue()


class CliNearbyTests(unittest.TestCase):
    def test_fixture_checks_the_fixtures_place(self):
        calls = []

        def fake(lat, lon):
            calls.append((lat, lon))
            return {t: (1 if t == 48098 else 300) for t in KNOWN_IDS}

        code, out, err = run(["--fixture", str(FIXTURE), "--json"], fake)
        self.assertEqual(code, 0, err)
        with open(FIXTURE, encoding="utf-8") as f:
            place = json.load(f)["place"]
        self.assertEqual(calls, [(place["lat"], place["lon"])])
        r = json.loads(out)
        self.assertTrue(r["nearby"]["available"])
        red = [s for s in r["species"] if s["taxon_id"] == 48098][0]
        self.assertEqual((red["nearby_observations"], red["rare_here"]), (1, True))
        self.assertNotIn("red maple", r["headline"])

    def test_text_card_lists_rare_last(self):
        code, out, _ = run(["--fixture", str(FIXTURE)], lambda lat, lon: {t: (0 if t == 48098 else 9) for t in KNOWN_IDS})
        self.assertEqual(code, 0)
        self.assertIn(card.RARE_LINE, " ".join(out.split()))
        others = [out.index(" %s (" % n) for n in ("sugar maple", "Norway maple", "sassafras")]
        self.assertGreater(out.index(" red maple ("), max(others))

    def test_check_failure_still_prints_the_card(self):
        def broken(lat, lon):
            raise OSError("offline")

        for fn in (lambda lat, lon: None, broken, None):
            code, out, err = run(["--fixture", str(FIXTURE)], fn)
            self.assertEqual(code, 0, err)
            self.assertIn(card.NEARBY_UNAVAILABLE_LINE, " ".join(out.split()))


if __name__ == "__main__":
    unittest.main()
