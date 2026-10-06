import unittest

from peakweek import config, inat


def ann(value, score=0, attr=36):
    return {"controlled_attribute_id": attr, "controlled_value_id": value, "vote_score": score}


def obs(**kw):
    base = {
        "id": 1, "observed_on": "2023-10-10", "location": "40.4862,-74.4518", "obscured": False,
        "geoprivacy": None, "taxon_geoprivacy": "open", "positional_accuracy": 15,
        "quality_grade": "research", "captive": False, "license_code": "cc-by-nc",
        "annotations": [ann(39)],
    }
    base.update(kw)
    return base


class TestAnnotationParsing(unittest.TestCase):
    def test_values_map_to_labels(self):
        self.assertEqual(inat.parse_leaves_label([ann(38)]), ("green", "ok"))
        self.assertEqual(inat.parse_leaves_label([ann(39)]), ("colored", "ok"))
        self.assertEqual(inat.parse_leaves_label([ann(40)]), ("bare", "ok"))

    def test_breaking_buds_ignored(self):
        self.assertEqual(inat.parse_leaves_label([ann(37)]), (None, "no_usable_leaves_value"))
        self.assertEqual(inat.parse_leaves_label([ann(37), ann(39)]), ("colored", "ok"))

    def test_other_attributes_ignored(self):
        self.assertEqual(inat.parse_leaves_label([ann(21, attr=12)]), (None, "no_leaves_annotation"))
        self.assertEqual(inat.parse_leaves_label([]), (None, "no_leaves_annotation"))
        self.assertEqual(inat.parse_leaves_label(None), (None, "no_leaves_annotation"))

    def test_vote_scores(self):
        # vote_score >= 0 counts; negative is ignored; missing score counts as 0.
        self.assertEqual(inat.parse_leaves_label([ann(39, score=0)]), ("colored", "ok"))
        self.assertEqual(inat.parse_leaves_label([ann(39, score=2)]), ("colored", "ok"))
        self.assertEqual(inat.parse_leaves_label([ann(39, score=-1)]), (None, "no_usable_leaves_value"))
        self.assertEqual(inat.parse_leaves_label([{"controlled_attribute_id": 36, "controlled_value_id": 40}]),
                         ("bare", "ok"))

    def test_conflicts(self):
        self.assertEqual(inat.parse_leaves_label([ann(38), ann(39)]), (None, "conflicting_leaves"))
        # A downvoted conflicting annotation does not count, so no conflict remains.
        self.assertEqual(inat.parse_leaves_label([ann(38, score=-2), ann(39, score=1)]), ("colored", "ok"))
        # Duplicate agreeing annotations are fine.
        self.assertEqual(inat.parse_leaves_label([ann(40), ann(40, score=3)]), ("bare", "ok"))


class TestCoordinatesAndQuality(unittest.TestCase):
    def test_clean_observation_kept(self):
        rec, why = inat.to_record(obs(), 48098)
        self.assertEqual(why, "ok")
        self.assertEqual(rec["label"], "colored")
        self.assertEqual(rec["lat"], 40.486)
        self.assertEqual(rec["lon"], -74.452)
        self.assertEqual(rec["taxon_id"], 48098)

    def test_obscured_dropped(self):
        self.assertEqual(inat.to_record(obs(obscured=True), 1)[1], "obscured")
        self.assertEqual(inat.to_record(obs(geoprivacy="obscured"), 1)[1], "obscured")
        self.assertEqual(inat.to_record(obs(geoprivacy="private"), 1)[1], "obscured")
        self.assertEqual(inat.to_record(obs(taxon_geoprivacy="obscured"), 1)[1], "obscured")

    def test_positional_accuracy(self):
        self.assertEqual(inat.to_record(obs(positional_accuracy=1500), 1)[1], "positional_accuracy_gt_1000m")
        self.assertEqual(inat.to_record(obs(positional_accuracy=1000), 1)[1], "ok")
        self.assertEqual(inat.to_record(obs(positional_accuracy=None), 1)[1], "ok")

    def test_missing_location(self):
        self.assertEqual(inat.to_record(obs(location=None), 1)[1], "no_location")

    def test_outside_region(self):
        self.assertEqual(inat.to_record(obs(location="35.0,-74.0"), 1)[1], "outside_region")
        self.assertEqual(inat.to_record(obs(location="44.0,-82.0"), 1)[1], "outside_region")

    def test_quality(self):
        self.assertEqual(inat.to_record(obs(quality_grade="casual"), 1)[1], "casual")
        self.assertEqual(inat.to_record(obs(captive=True, quality_grade="casual"), 1)[1], "captive_or_cultivated")
        self.assertEqual(inat.to_record(obs(quality_grade="needs_id"), 1)[1], "ok")


class TestRegion(unittest.TestCase):
    def test_bounds_inclusive(self):
        r = config.REGION
        self.assertTrue(config.in_region(r["lat_min"], r["lon_min"]))
        self.assertTrue(config.in_region(r["lat_max"], r["lon_max"]))
        self.assertTrue(config.in_region(40.4862, -74.4518))   # New Brunswick NJ
        self.assertTrue(config.in_region(44.4759, -73.2121))   # Burlington VT
        self.assertTrue(config.in_region(40.4406, -79.9959))   # Pittsburgh PA

    def test_outside(self):
        self.assertFalse(config.in_region(38.49, -75.0))
        self.assertFalse(config.in_region(47.51, -70.0))
        self.assertFalse(config.in_region(42.0, -80.51))
        self.assertFalse(config.in_region(42.0, -66.89))
        self.assertFalse(config.in_region(33.749, -84.388))    # Atlanta


if __name__ == "__main__":
    unittest.main()
