import datetime as dt
import json
import sys
import unittest

import numpy as np

from peakweek import features, forecast
from peakweek.species import SPECIES
from tests.helpers import FakeClassifier, synthetic_forecast_weather, synthetic_pool

TODAY = dt.date(2026, 10, 6)


def run(lat=40.4862, lon=-74.4518, name="New Brunswick, NJ", **kw):
    fake = FakeClassifier()
    out = forecast.forecast(
        lat, lon, name, TODAY, classifier=fake, weather_daily=synthetic_forecast_weather(TODAY),
        elevation_m=20.0, context=synthetic_pool(2500, seed=5),
        clim_curve=np.full(features.N_SEASON_OFFSETS, 16.0), **kw)
    return out, fake


class TestForecastContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out, cls.fake = run()

    def test_top_level_keys(self):
        for k in ("place", "in_region", "region_note", "generated_at", "today", "days", "model", "weather", "species"):
            self.assertIn(k, self.out)
        self.assertEqual(self.out["place"]["name"], "New Brunswick, NJ")
        self.assertEqual(self.out["place"]["elevation_m"], 20.0)
        self.assertEqual(self.out["today"], "2026-10-06")
        self.assertTrue(self.out["generated_at"].endswith("Z"))
        self.assertEqual(self.out["weather"]["past_days"], 92)
        self.assertEqual(self.out["weather"]["forecast_days"], 16)
        for k in ("name", "context_rows", "seconds", "training_data"):
            self.assertIn(k, self.out["model"])

    def test_days(self):
        days = self.out["days"]
        self.assertEqual(len(days), 15)
        self.assertEqual(days[0], "2026-10-06")
        self.assertEqual(days[-1], "2026-10-20")
        self.assertEqual(days, sorted(days))

    def test_species_and_probabilities(self):
        self.assertEqual([s["taxon_id"] for s in self.out["species"]], [s["taxon_id"] for s in SPECIES])
        for s in self.out["species"]:
            self.assertEqual(set(s) >= {"taxon_id", "common", "scientific", "daily"}, True)
            self.assertEqual([d["date"] for d in s["daily"]], self.out["days"])
            for d in s["daily"]:
                total = d["green"] + d["colored"] + d["bare"]
                self.assertAlmostEqual(total, 1.0, delta=1e-6)
                for k in ("green", "colored", "bare"):
                    self.assertGreaterEqual(d[k], 0.0)
                    self.assertLessEqual(d[k], 1.0)

    def test_json_serializable(self):
        json.loads(json.dumps(self.out))

    def test_in_region(self):
        self.assertTrue(self.out["in_region"])
        self.assertNotIn("extrapolation", self.out["region_note"])

    def test_outside_region_still_returns_probabilities(self):
        out, _ = run(lat=33.749, lon=-84.388, name="Atlanta, GA")
        self.assertFalse(out["in_region"])
        self.assertIn("extrapolation", out["region_note"])
        self.assertEqual(len(out["species"]), len(SPECIES))

    def test_no_tabpfn_import(self):
        self.assertNotIn("tabpfn", sys.modules)

    def test_query_features_finite_for_horizon(self):
        rows = forecast.query_rows(40.4862, -74.4518, 20.0,
                                   [TODAY + dt.timedelta(days=i) for i in range(15)],
                                   synthetic_forecast_weather(TODAY), np.full(features.N_SEASON_OFFSETS, 16.0))
        self.assertEqual(len(rows), 15 * len(SPECIES))
        for c in features.FEATURES_FULL:
            self.assertFalse(rows[c].isna().any(), c)

    def test_locked_config_matches_locked_json(self):
        from peakweek.config import ROOT
        path = ROOT / "eval" / "locked.json"
        if not path.exists():
            self.skipTest("eval/locked.json not written yet")
        self.assertEqual(json.loads(path.read_text())["tabpfn_runs"]["tabpfn"], forecast.LOCKED_CONFIG)

    def test_one_fit_per_subsample(self):
        # stratified context is location-independent: one fit per averaged context, 120 rows each
        self.assertEqual(self.out["model"]["n_subsamples"], forecast.LOCKED_CONFIG["n_subsamples"])
        self.assertEqual(len(self.fake.fit_sizes), forecast.LOCKED_CONFIG["n_subsamples"])
        self.assertTrue(all(n <= forecast.LOCKED_CONFIG["n_context"] for n in self.fake.fit_sizes))


if __name__ == "__main__":
    unittest.main()
