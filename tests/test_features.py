import datetime as dt
import math
import unittest

import numpy as np
import pandas as pd

from peakweek import features


def series(start, values, col="tmean"):
    d0 = pd.Timestamp(start)
    return pd.DataFrame({"date": [d0 + pd.Timedelta(days=i) for i in range(len(values))], col: values})


def weather_frame(start, tmean, tmin=None, prcp=None):
    n = len(tmean)
    d0 = pd.Timestamp(start)
    return pd.DataFrame({
        "date": [d0 + pd.Timedelta(days=i) for i in range(n)],
        "tmean": tmean,
        "tmin": tmin if tmin is not None else [np.nan] * n,
        "prcp": prcp if prcp is not None else [0.0] * n,
    })


class TestDaylength(unittest.TestCase):
    # Published sunrise-sunset day lengths (top of disk, standard refraction).
    def test_new_york_solstices(self):
        # New York City (40.71 N): ~15 h 05 m on the June solstice, ~9 h 15 m on the December solstice.
        self.assertAlmostEqual(features.daylength_hours(40.71, 172), 15 + 5 / 60, delta=0.1)
        self.assertAlmostEqual(features.daylength_hours(40.71, 355), 9 + 15 / 60, delta=0.1)

    def test_equator_and_equinox(self):
        # Equator: ~12 h 07 m all year. 40.7 N at the September equinox: ~12 h 10 m.
        for doy in (1, 100, 200, 300):
            self.assertAlmostEqual(features.daylength_hours(0.0, doy), 12 + 7 / 60, delta=0.1)
        self.assertAlmostEqual(features.daylength_hours(40.71, 265), 12 + 10 / 60, delta=0.12)

    def test_polar_clipping(self):
        self.assertAlmostEqual(features.daylength_hours(80.0, 172), 24.0, places=6)
        self.assertAlmostEqual(features.daylength_hours(80.0, 355), 0.0, places=6)

    def test_october_shortens(self):
        a = features.daylength_hours(44.0, 275)
        b = features.daylength_hours(44.0, 290)
        self.assertLess(b, a)
        self.assertTrue(10.0 < b < 12.0)


class TestAccumulators(unittest.TestCase):
    def test_chill_degree_days(self):
        self.assertEqual(features.chill_degree_days(np.array([25.0, 20.0, 15.0, 10.0])), 0 + 0 + 5 + 10)
        self.assertEqual(features.chill_degree_days(np.array([])), 0.0)
        self.assertTrue(math.isnan(features.chill_degree_days(np.array([15.0, np.nan]))))

    def test_frost_counts(self):
        tmin = np.array([6.0, 5.0, 0.5, 0.0, -2.0])
        self.assertEqual(features.count_nights_at_or_below(tmin, 0.0), 2)
        self.assertEqual(features.count_nights_at_or_below(tmin, 5.0), 4)
        self.assertTrue(math.isnan(features.count_nights_at_or_below(np.array([1.0, np.nan]), 0.0)))

    def test_weather_features_window_ends_day_before(self):
        # Sep 1..Sep 10 2023; tmean = 10 + day index, tmin = day index - 3, prcp = 2 mm
        n = 10
        wx = weather_frame("2023-09-01", [10.0 + i for i in range(n)], [i - 3.0 for i in range(n)], [2.0] * n)
        prep = features.prepare_daily(wx)
        f = features.weather_features_on(prep, dt.date(2023, 9, 6))  # uses Sep 1..Sep 5 (i = 0..4)
        self.assertAlmostEqual(f["cdd20"], sum(20 - (10 + i) for i in range(5)))
        self.assertEqual(f["frost0"], 4)   # tmin -3,-2,-1,0
        self.assertEqual(f["frost5"], 5)   # all of -3..1
        self.assertAlmostEqual(f["tmin7"], np.mean([i - 3.0 for i in range(5)]))   # window clipped at Sep 1
        self.assertAlmostEqual(f["tmean14"], np.mean([10.0 + i for i in range(5)]))
        self.assertAlmostEqual(f["prcp30"], 2.0)
        self.assertAlmostEqual(f["tmean_season"], 12.0)

    def test_season_start_day_has_empty_windows(self):
        wx = weather_frame("2023-08-20", [15.0] * 20, [5.0] * 20)
        f = features.weather_features_on(features.prepare_daily(wx), dt.date(2023, 9, 1))
        self.assertEqual(f["cdd20"], 0.0)
        self.assertEqual(f["frost0"], 0.0)
        self.assertTrue(math.isnan(f["tmin7"]))
        self.assertTrue(math.isnan(f["tmean_season"]))

    def test_accumulators_ignore_august(self):
        # August is very cold in this fake series but must not count: accumulators start Sep 1.
        tmean = [-5.0] * 31 + [25.0] * 10
        tmin = [-10.0] * 31 + [15.0] * 10
        wx = weather_frame("2023-08-01", tmean, tmin)
        f = features.weather_features_on(features.prepare_daily(wx), dt.date(2023, 9, 8))
        self.assertEqual(f["cdd20"], 0.0)
        self.assertEqual(f["frost0"], 0.0)
        self.assertAlmostEqual(f["tmin7"], 15.0)

    def test_tmin7_uses_last_seven_days(self):
        tmin = [float(i) for i in range(30)]
        wx = weather_frame("2023-09-01", [10.0] * 30, tmin)
        f = features.weather_features_on(features.prepare_daily(wx), dt.date(2023, 9, 21))  # days 13..19
        self.assertAlmostEqual(f["tmin7"], np.mean(range(13, 20)))


class TestMissingDays(unittest.TestCase):
    def test_short_gap_interpolated(self):
        tmean = [10.0, np.nan, np.nan, 16.0, 18.0]
        prep = features.prepare_daily(weather_frame("2023-09-01", tmean, [1.0] * 5))
        self.assertEqual(list(prep["tmean"].round(6)), [10.0, 12.0, 14.0, 16.0, 18.0])

    def test_long_gap_makes_sums_nan_but_means_survive(self):
        tmean = [10.0] * 5 + [np.nan] * 3 + [10.0] * 12
        wx = weather_frame("2023-09-01", tmean, [3.0] * 20)
        prep = features.prepare_daily(wx)
        self.assertEqual(int(prep["tmean"].isna().sum()), 3)  # 3-day interior gap is not filled
        f = features.weather_features_on(prep, dt.date(2023, 9, 21))
        self.assertTrue(math.isnan(f["cdd20"]))
        self.assertAlmostEqual(f["tmean14"], 10.0)
        self.assertEqual(f["frost5"], 20.0)

    def test_dropped_dates_are_reindexed(self):
        wx = weather_frame("2023-09-01", [10.0] * 10, [2.0] * 10)
        wx = wx.drop(index=[4]).reset_index(drop=True)  # Sep 5 row missing entirely
        prep = features.prepare_daily(wx)
        self.assertEqual(len(prep), 10)
        self.assertAlmostEqual(prep.loc[pd.Timestamp("2023-09-05"), "tmean"], 10.0)

    def test_trailing_gap_forward_filled_up_to_two_days(self):
        tmean = [10.0, 11.0, np.nan, np.nan, np.nan]
        prep = features.prepare_daily(weather_frame("2023-09-01", tmean, [1.0] * 5))
        self.assertEqual(prep["tmean"].tolist()[:4], [10.0, 11.0, 11.0, 11.0])
        self.assertTrue(math.isnan(prep["tmean"].tolist()[4]))

    def test_window_beyond_data_is_missing(self):
        wx = weather_frame("2023-09-01", [10.0] * 5, [1.0] * 5)
        f = features.weather_features_on(features.prepare_daily(wx), dt.date(2023, 9, 20))
        self.assertTrue(math.isnan(f["cdd20"]))
        self.assertTrue(math.isnan(f["tmin7"]))

    def test_precip_half_window_rule(self):
        prcp = [1.0] * 10 + [np.nan] * 20
        wx = weather_frame("2023-09-01", [10.0] * 30, [1.0] * 30, prcp)
        f = features.weather_features_on(features.prepare_daily(wx), dt.date(2023, 10, 1))
        self.assertTrue(math.isnan(f["prcp30"]))  # only 10/30 present
        f2 = features.weather_features_on(features.prepare_daily(wx), dt.date(2023, 9, 15))
        self.assertAlmostEqual(f2["prcp30"], 1.0)  # Sep 1..14: 10 of 14 present


class TestClimatology(unittest.TestCase):
    def test_leave_one_out(self):
        curves = [np.array([np.nan, 1.0, 2.0]), np.array([np.nan, 3.0, np.nan]), np.array([np.nan, 5.0, 4.0])]
        total, count = features.climatology_from_curves(curves)
        self.assertEqual(count.tolist(), [0, 3, 2])
        allm = features.leave_one_out_mean(total, count, None)
        self.assertTrue(math.isnan(allm[0]))
        self.assertAlmostEqual(allm[1], 3.0)
        loo = features.leave_one_out_mean(total, count, curves[0])
        self.assertAlmostEqual(loo[1], 4.0)
        self.assertAlmostEqual(loo[2], 4.0)
        loo2 = features.leave_one_out_mean(total, count, curves[1])  # NaN own value: nothing removed
        self.assertAlmostEqual(loo2[2], 3.0)

    def test_season_curve_matches_weather_features(self):
        rng = np.random.default_rng(1)
        tmean = list(15 + rng.normal(size=91))
        wx = weather_frame("2023-09-01", tmean, [5.0] * 91)
        prep = features.prepare_daily(wx)
        curve = features.season_mean_curve(prep, 2023)
        for k in (1, 10, 45, 90):
            d = dt.date(2023, 9, 1) + dt.timedelta(days=k)
            self.assertAlmostEqual(curve[k], features.weather_features_on(prep, d)["tmean_season"])

    def test_feature_rows_anomaly(self):
        wx = weather_frame("2023-09-01", [12.0] * 60, [4.0] * 60)
        pts = pd.DataFrame({"taxon_id": [48098, 49005], "date": ["2023-09-21", "2023-10-01"],
                            "lat": [40.5, 41.0], "lon": [-74.4, -74.0], "elevation_m": [20.0, 100.0]})
        clim = np.full(features.N_SEASON_OFFSETS, 10.0)
        rows = features.feature_rows(pts, wx, clim)
        self.assertEqual(list(rows["tanom"].round(6)), [2.0, 2.0])
        self.assertEqual(list(rows["doy"]), [264, 274])
        self.assertEqual(list(rows["species_code"]), [0.0, 3.0])
        for c in features.FEATURES_FULL:
            self.assertIn(c, rows.columns)


class TestMergeHistory(unittest.TestCase):
    def test_archive_preferred_forecast_fills(self):
        from peakweek import weather
        ar = weather_frame("2026-09-01", [10.0, 11.0, np.nan], [1.0, 2.0, np.nan], [5.0, 0.0, np.nan])
        fc = weather_frame("2026-09-02", [20.0, 21.0, 22.0], [9.0, 9.0, 9.0], [0.0, 0.0, 1.0])
        m = weather.merge_history(ar, fc)
        self.assertEqual(list(m["date"].dt.day), [1, 2, 3, 4])
        self.assertEqual(list(m["tmean"]), [10.0, 11.0, 21.0, 22.0])   # archive wins on Sep 2; forecast fills 3-4
        self.assertEqual(list(m["prcp"]), [5.0, 0.0, 0.0, 1.0])


class TestOfflineGuard(unittest.TestCase):
    def test_cache_miss_raises_when_offline(self):
        import os
        import tempfile
        from pathlib import Path
        from peakweek import weather
        old = os.environ.get("PEAKWEEK_OFFLINE")
        os.environ["PEAKWEEK_OFFLINE"] = "1"
        try:
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(weather.OfflineError):
                    weather._get_json_cached("https://example.invalid/x", {"a": 1}, Path(d) / "missing.json", 1.0, "t")
                hit = Path(d) / "hit.json"
                hit.write_text('{"ok": true}')
                self.assertEqual(weather._get_json_cached("https://example.invalid/x", {}, hit, 1.0, "t"), {"ok": True})
        finally:
            if old is None:
                os.environ.pop("PEAKWEEK_OFFLINE", None)
            else:
                os.environ["PEAKWEEK_OFFLINE"] = old


if __name__ == "__main__":
    unittest.main()
