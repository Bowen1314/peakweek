import unittest

import numpy as np
import pandas as pd

from peakweek import config, dataset, features, weather


class TestCells(unittest.TestCase):
    def test_cell_center_and_id(self):
        deg = config.GRID_DEG
        lat, lon = 40.4862, -74.4518
        clat, clon = weather.cell_center(lat, lon)
        self.assertLessEqual(abs(clat - lat), deg / 2)
        self.assertLessEqual(abs(clon - lon), deg / 2)
        self.assertEqual(weather.cell_id(lat, lon), f"{clat:.2f}_{clon:.2f}")

    def test_assign_cells_weather_point_is_centroid(self):
        df = pd.DataFrame({"obs_id": [1, 2, 3], "observed_on": ["2023-10-01", "2024-10-02", "2024-10-03"],
                           "lat": [40.40, 40.60, 44.48], "lon": [-74.40, -74.20, -73.21]})
        out = dataset.assign_cells(df)
        a = out.iloc[0]
        self.assertEqual(out.iloc[0]["cell_id"], out.iloc[1]["cell_id"])
        self.assertAlmostEqual(a["cell_lat"], 40.5)
        self.assertAlmostEqual(a["cell_lon"], -74.3)
        self.assertEqual(list(out["year"]), [2023, 2024, 2024])

    def test_weather_point_fallback(self):
        clim = pd.DataFrame({"cell_id": ["40.50_-74.50"], "cell_lat": [40.42], "cell_lon": [-74.33],
                             "offset": [0], "clim_tmean_season": [np.nan], "n_years": [0]})
        lat, lon, cid, known = dataset.weather_point(40.4862, -74.4518, clim)
        if cid == "40.50_-74.50":
            self.assertTrue(known)
            self.assertEqual((lat, lon), (40.42, -74.33))
        lat, lon, cid, known = dataset.weather_point(33.75, -84.39, clim)
        self.assertFalse(known)
        self.assertEqual((lat, lon), weather.cell_center(33.75, -84.39))

    def test_weighted_cost(self):
        self.assertEqual(weather.weighted_cost(91, 3), 6.5)
        self.assertEqual(weather.weighted_cost(7, 3), 1.0)
        self.assertEqual(weather.weighted_cost(108, 3), 108 / 14)
        self.assertEqual(weather.weighted_cost(14, 15), 1.5)

    def test_split_years(self):
        df = pd.DataFrame({"year": [2018, 2023, 2024, 2025], "x": range(4)})
        tr, te = dataset.split(df, *dataset.SPLITS["val"])
        self.assertEqual(list(tr["year"]), [2018, 2023])
        self.assertEqual(list(te["year"]), [2024])
        tr, te = dataset.split(df, *dataset.SPLITS["test"])
        self.assertEqual(list(tr["year"]), [2018, 2023, 2024])
        self.assertEqual(list(te["year"]), [2025])

    def test_climatology_curve_roundtrip(self):
        curves = {2020: np.linspace(10, 20, features.N_SEASON_OFFSETS), 2021: np.linspace(12, 22, features.N_SEASON_OFFSETS)}
        rows = dataset.climatology_rows("40.50_-74.50", 40.4, -74.3, curves)
        curve = dataset.climatology_curve(rows, "40.50_-74.50")
        np.testing.assert_allclose(curve, np.linspace(11, 21, features.N_SEASON_OFFSETS), atol=1e-4)
        self.assertIsNone(dataset.climatology_curve(rows, "nope"))


if __name__ == "__main__":
    unittest.main()
