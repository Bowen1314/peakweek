import copy
import datetime as dt
import json
import unittest
from pathlib import Path

from peakweek import verdict

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "forecast_fixture.json"


def make_result(start, rows_by_species, in_region=True):
    """rows_by_species: {common: [(green, colored, bare), ...]} -> a forecast()-shaped dict."""
    start = dt.date.fromisoformat(start)
    n = len(next(iter(rows_by_species.values())))
    days = [(start + dt.timedelta(days=i)).isoformat() for i in range(n)]
    species = []
    for k, (common, rows) in enumerate(rows_by_species.items()):
        species.append({
            "taxon_id": 1000 + k, "common": common, "scientific": "Testus %s" % k,
            "daily": [{"date": d, "green": g, "colored": c, "bare": b} for d, (g, c, b) in zip(days, rows)],
        })
    return {"place": {"name": "Testville", "lat": 40.0, "lon": -74.0}, "in_region": in_region,
            "region_note": "Trained on somewhere.", "today": days[0], "days": days, "species": species}


def flat(g, c, b, n=15):
    return [(g, c, b)] * n


def load_fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


class WeekendTests(unittest.TestCase):
    def test_midweek_takes_next_saturday_and_sunday(self):
        a = verdict.annotate(make_result("2026-10-06", {"a": flat(.5, .4, .1)}))  # Tuesday
        self.assertEqual(a["weekend_dates"], ["2026-10-10", "2026-10-11"])

    def test_today_saturday_is_this_weekend(self):
        a = verdict.annotate(make_result("2026-10-10", {"a": flat(.5, .4, .1)}))
        self.assertEqual(a["weekend_dates"], ["2026-10-10", "2026-10-11"])

    def test_today_sunday_is_just_today(self):
        a = verdict.annotate(make_result("2026-10-11", {"a": flat(.5, .4, .1)}))
        self.assertEqual(a["weekend_dates"], ["2026-10-11"])

    def test_weekend_partly_outside_days_uses_saturday_only(self):
        rows = [(.8, .1, .1)] * 4 + [(.2, .7, .1)]  # Tue..Sat; Sunday not in days
        a = verdict.annotate(make_result("2026-10-06", {"a": rows}))
        self.assertEqual(a["weekend_dates"], ["2026-10-10"])
        self.assertAlmostEqual(a["species"][0]["weekend"]["colored"], .7)

    def test_no_weekend_day_raises(self):
        with self.assertRaises(ValueError):
            verdict.annotate(make_result("2026-10-05", {"a": [(.5, .4, .1)] * 5}))  # Mon..Fri

    def test_weekend_is_mean_of_saturday_and_sunday(self):
        rows = [(.9, .1, 0.)] * 4 + [(.4, .5, .1), (.2, .6, .2)] + [(.1, .5, .4)] * 9
        sp = verdict.annotate(make_result("2026-10-06", {"a": rows}))["species"][0]
        self.assertAlmostEqual(sp["weekend"]["green"], .3)
        self.assertAlmostEqual(sp["weekend"]["colored"], .55)
        self.assertAlmostEqual(sp["weekend"]["bare"], .15)


class BestDayTests(unittest.TestCase):
    def test_best_day_is_argmax_colored(self):
        rows = [(.8, .2, 0.)] * 15
        rows[9] = (.2, .7, .1)
        sp = verdict.annotate(make_result("2026-10-06", {"a": rows}))["species"][0]
        self.assertEqual(sp["best_day"], "2026-10-15")
        self.assertEqual(sp["best_colored"], .7)

    def test_tie_goes_to_earliest_day(self):
        rows = [(.8, .2, 0.)] * 15
        rows[3] = rows[7] = rows[12] = (.3, .6, .1)
        sp = verdict.annotate(make_result("2026-10-06", {"a": rows}))["species"][0]
        self.assertEqual(sp["best_day"], "2026-10-09")

    def test_all_equal_best_is_today(self):
        sp = verdict.annotate(make_result("2026-10-06", {"a": flat(.5, .3, .2)}))["species"][0]
        self.assertEqual(sp["best_day"], "2026-10-06")


class VerdictRuleTests(unittest.TestCase):
    def verdict_of(self, rows, start="2026-10-06"):
        sp = verdict.annotate(make_result(start, {"a": rows}))["species"][0]
        return sp["verdict"], sp["verdict_text"]

    def test_grid_every_case_gets_exactly_the_documented_verdict(self):
        # Flat series: best day = today and nothing falls, so the peak-behind rule never fires.
        for ci in range(0, 101, 5):
            for bi in range(0, 101 - ci, 5):
                c, b = ci / 100, bi / 100
                g = round(1 - c - b, 10)
                v, text = self.verdict_of(flat(g, c, b))
                if b >= verdict.LATE_BARE:
                    want = "late"
                elif c >= verdict.GO_COLORED:
                    want = "go"
                elif c >= verdict.START_COLORED:
                    want = "starting"
                else:
                    want = "wait"
                self.assertEqual(v, want, (c, b))
                self.assertIn(v, verdict.VERDICTS)
                self.assertTrue(text)

    def test_high_color_with_some_bare_is_go_not_wait(self):
        v, text = self.verdict_of(flat(.1, .6, .3))
        self.assertEqual(v, "go")
        self.assertIn("leaves are starting to fall", text)

    def test_go_hints_at_a_much_better_day_after_the_weekend(self):
        rows = [(.45, .55, 0.)] * 15
        rows[10] = (.2, .8, 0.)
        v, text = self.verdict_of(rows)
        self.assertEqual(v, "go")
        self.assertIn("even more color around Fri, Oct 16", text)

    def test_peak_behind_is_late(self):
        # Best day today (Tue), color falling and bare rising toward the weekend.
        rows = [(.2, .45 - .02 * i, .35 + .02 * i - .2) for i in range(15)]
        rows = [(round(1 - c - b, 6), c, b) for _, c, b in rows]
        v, text = self.verdict_of(rows)
        self.assertEqual(v, "late")
        self.assertEqual(text, "Color is fading; go soon")

    def test_falling_but_still_colorful_is_go(self):
        rows = [(round(1 - (.75 - .01 * i) - (.05 + .01 * i), 6), .75 - .01 * i, .05 + .01 * i) for i in range(15)]
        self.assertEqual(self.verdict_of(rows)[0], "go")

    def test_peak_behind_when_today_is_saturday(self):
        rows = [(.3, .45, .25), (.3, .38, .32)] + [(.3, .3, .4)] * 13
        self.assertEqual(self.verdict_of(rows, start="2026-10-10")[0], "late")

    def test_starting_names_best_day(self):
        rows = [(.7 - .03 * i, .3 + .03 * i, 0.) for i in range(15)]
        rows[13] = (.2, .8, 0.)
        rows[14] = (.25, .75, 0.)
        v, text = self.verdict_of(rows)
        self.assertEqual(v, "starting")
        self.assertIn("best around Mon, Oct 19", text)

    def test_wait_rising_at_end_says_or_later(self):
        rows = [(.95 - .03 * i, .05 + .03 * i, 0.) for i in range(15)]
        v, text = self.verdict_of(rows)
        self.assertEqual(v, "wait")
        self.assertIn("best around Tue, Oct 20 or later", text)

    def test_wait_without_color_in_window(self):
        v, text = self.verdict_of(flat(.95, .03, .02))
        self.assertEqual((v, text), ("wait", "Mostly green through Tue, Oct 20"))

    def test_many_bare_is_late(self):
        self.assertEqual(self.verdict_of(flat(.1, .5, .4)), ("late", "Many leaves are already down; go soon"))

    def test_nothing_left_does_not_say_go(self):
        self.assertEqual(self.verdict_of(flat(0., .1, .9)), ("late", "Most leaves are already down"))


class AnnotateTests(unittest.TestCase):
    def test_probabilities_untouched_and_input_not_mutated(self):
        raw = load_fixture()
        before = copy.deepcopy(raw)
        a = verdict.annotate(raw)
        self.assertEqual(raw, before)
        for sa, sb in zip(a["species"], before["species"]):
            self.assertEqual(sa["daily"], sb["daily"])
        self.assertEqual(a["days"], before["days"])

    def test_weekend_probabilities_sum_to_one(self):
        a = verdict.annotate(load_fixture())
        for sp in a["species"]:
            self.assertAlmostEqual(sum(sp["weekend"].values()), 1.0, delta=1e-6)
            for row in sp["daily"]:
                self.assertAlmostEqual(row["green"] + row["colored"] + row["bare"], 1.0, delta=1e-6)

    def test_contract_fields_present(self):
        a = verdict.annotate(load_fixture())
        self.assertTrue(a["headline"])
        self.assertEqual(a["weekend_dates"], ["2026-10-10", "2026-10-11"])
        for sp in a["species"]:
            for key in ("weekend", "best_day", "best_colored", "verdict", "verdict_text"):
                self.assertIn(key, sp)
            self.assertIn(sp["verdict"], verdict.VERDICTS)
            self.assertIn(sp["best_day"], a["days"])

    def test_fixture_has_15_days_and_is_marked_synthetic(self):
        raw = load_fixture()
        self.assertIs(raw["synthetic"], True)
        self.assertEqual(len(raw["days"]), 15)

    def test_mismatched_dates_raise(self):
        r = make_result("2026-10-06", {"a": flat(.5, .4, .1)})
        r["species"][0]["daily"][3]["date"] = "2026-01-01"
        with self.assertRaises(ValueError):
            verdict.annotate(r)

    def test_wrong_length_raises(self):
        r = make_result("2026-10-06", {"a": flat(.5, .4, .1)})
        r["species"][0]["daily"].pop()
        with self.assertRaises(ValueError):
            verdict.annotate(r)

    def test_idempotent(self):
        a = verdict.annotate(load_fixture())
        self.assertEqual(verdict.annotate(a), a)


class HeadlineTests(unittest.TestCase):
    def test_fixture_headline(self):
        a = verdict.annotate(load_fixture())
        self.assertEqual(a["headline"],
                         "This weekend, go see red maples and sassafras; come back around Sat, Oct 17 for sugar maples.")

    def test_at_most_three_go_species_most_colorful_first(self):
        r = make_result("2026-10-06", {
            "red maple": flat(.3, .6, .1), "sweetgum": flat(.2, .7, .1), "black gum": flat(.15, .8, .05),
            "sugar maple": flat(.35, .55, .1)})
        h = verdict.annotate(r)["headline"]
        self.assertEqual(h, "This weekend, go see black gums, sweetgums and red maples.")

    def test_no_go_but_starting(self):
        rising = [(.65 - .03 * i, .35 + .03 * i, 0.) for i in range(15)]
        r = make_result("2026-10-06", {"red maple": rising, "American beech": flat(.95, .03, .02)})
        h = verdict.annotate(r)["headline"]
        self.assertTrue(h.startswith(
            "No tree here is in full color yet this weekend, but red maples are starting to turn"), h)

    def test_all_green(self):
        r = make_result("2026-10-06", {"red maple": flat(.95, .03, .02), "Norway maple": flat(.97, .02, .01)})
        h = verdict.annotate(r)["headline"]
        self.assertEqual(h, "Still mostly green this weekend, and little color is expected through Tue, Oct 20.")

    def test_all_green_but_one_turning_later(self):
        rising = [(.98 - .03 * i, .02 + .03 * i, 0.) for i in range(15)]  # weekend ~.16, Oct 20 ~.44
        r = make_result("2026-10-06", {"sweetgum": rising, "Norway maple": flat(.97, .02, .01)})
        h = verdict.annotate(r)["headline"]
        self.assertEqual(h, "Still mostly green this weekend; come back around Tue, Oct 20 for sweetgums.")

    def test_all_late_with_some_color(self):
        r = make_result("2026-10-06", {"black gum": flat(.05, .5, .45), "sassafras": flat(.05, .3, .65)})
        h = verdict.annotate(r)["headline"]
        self.assertEqual(h, "Color is fading this weekend; black gums and sassafras still show some color.")

    def test_all_bare(self):
        r = make_result("2026-10-06", {"black gum": flat(.0, .1, .9)})
        self.assertEqual(verdict.annotate(r)["headline"], "Most leaves are already down this weekend.")

    def test_out_of_region_is_hedged(self):
        r = make_result("2026-10-06", {"red maple": flat(.3, .6, .1)}, in_region=False)
        h = verdict.annotate(r)["headline"]
        self.assertTrue(h.startswith("Rough guide only (outside the area the model was trained on): this weekend"), h)


class NearbyTests(unittest.TestCase):
    """verdict.annotate(result, nearby): species rarely recorded within 50 km."""

    def result(self):
        # taxon ids 1000.. in this order: red maple, sweetgum, black gum, sugar maple
        rising = [(.98 - .03 * i, .02 + .03 * i, 0.) for i in range(15)]
        return make_result("2026-10-06", {
            "red maple": flat(.3, .6, .1), "sweetgum": flat(.2, .7, .1), "black gum": flat(.15, .8, .05),
            "sugar maple": rising})

    def test_counts_and_rare_flag(self):
        a = verdict.annotate(self.result(), {1000: 250, 1001: 4, 1002: 5})
        sp = {s["common"]: s for s in a["species"]}
        self.assertEqual((sp["red maple"]["nearby_observations"], sp["red maple"]["rare_here"]), (250, False))
        self.assertEqual((sp["sweetgum"]["nearby_observations"], sp["sweetgum"]["rare_here"]), (4, True))
        self.assertEqual((sp["black gum"]["nearby_observations"], sp["black gum"]["rare_here"]), (5, False))
        self.assertEqual((sp["sugar maple"]["nearby_observations"], sp["sugar maple"]["rare_here"]), (0, True))
        self.assertEqual(a["nearby"], {"radius_km": 50, "source": "iNaturalist research-grade observations",
                                       "available": True})

    def test_rare_text_prefix_and_verdict_kept(self):
        a = verdict.annotate(self.result(), {1000: 250, 1002: 99})
        sp = {s["common"]: s for s in a["species"]}
        self.assertEqual(sp["sweetgum"]["verdict"], "go")
        self.assertEqual(sp["sweetgum"]["verdict_text"], "Rarely recorded within 50 km; go this weekend")
        self.assertEqual(sp["red maple"]["verdict_text"], "Go this weekend")

    def test_rare_never_named_in_headline(self):
        # sweetgum is the most colorful but rare; sugar maple (the only "come back") is rare too
        a = verdict.annotate(self.result(), {1000: 250, 1002: 99})
        self.assertEqual(a["headline"], "This weekend, go see black gums and red maples.")
        self.assertNotIn("sweetgum", a["headline"])
        self.assertNotIn("sugar maple", a["headline"])
        b = verdict.annotate(self.result(), {1000: 250, 1001: 250, 1002: 99, 1003: 250})
        self.assertEqual(b["headline"], "This weekend, go see black gums, sweetgums and red maples; "
                                        "come back around Tue, Oct 20 for sugar maples.")

    def test_rare_does_not_drive_summary_sentence(self):
        # the only colorful tree is rare: the headline speaks for the common (green) ones
        r = make_result("2026-10-06", {"sweetgum": flat(.2, .7, .1), "Norway maple": flat(.97, .02, .01)})
        a = verdict.annotate(r, {1000: 1, 1001: 400})
        self.assertEqual(a["headline"], "Still mostly green this weekend, and little color is expected through Tue, Oct 20.")

    def test_all_rare(self):
        a = verdict.annotate(self.result(), {})
        self.assertTrue(all(s["rare_here"] for s in a["species"]))
        self.assertEqual(a["headline"], "None of these trees is commonly recorded within 50 km of here.")
        r = self.result()
        r["in_region"] = False
        self.assertEqual(verdict.annotate(r, {})["headline"],
                         "Rough guide only (outside the area the model was trained on): "
                         "none of these trees is commonly recorded within 50 km of here.")

    def test_unavailable(self):
        plain = verdict.annotate(self.result())
        self.assertEqual(plain["nearby"]["available"], False)
        for sp in plain["species"]:
            self.assertNotIn("rare_here", sp)
            self.assertNotIn("nearby_observations", sp)
        self.assertEqual(plain["headline"], "This weekend, go see black gums, sweetgums and red maples; "
                                            "come back around Tue, Oct 20 for sugar maples.")

    def test_probabilities_untouched_and_reannotate_is_clean(self):
        r = self.result()
        before = copy.deepcopy(r)
        a = verdict.annotate(r, {1000: 250})
        self.assertEqual(r, before)
        for sp, orig in zip(a["species"], before["species"]):
            self.assertEqual(sp["daily"], orig["daily"])
        self.assertEqual(verdict.annotate(a, {1000: 250}), a)  # prefix added once, not twice
        again = verdict.annotate(a)  # a later annotate without counts drops the old marks
        self.assertFalse(any("rare_here" in s for s in again["species"]))
        self.assertEqual(again["headline"], verdict.annotate(r)["headline"])

    def test_string_keys_from_json(self):
        a = verdict.annotate(self.result(), {"1000": 250, "1001": 3})
        self.assertEqual([s["rare_here"] for s in a["species"]], [False, True, True, True])

    def test_real_fixtures_with_saved_counts(self):
        # counts saved from iNaturalist for New Brunswick, NJ: all 8 trees are common there
        with open(ROOT / "tests" / "data" / "inat_species_counts_new_brunswick.json", encoding="utf-8") as f:
            sample = json.load(f)
        from peakweek import nearby
        ids = [s["taxon_id"] for s in load_fixture()["species"]]
        counts = nearby.parse_response(sample, ids)
        a = verdict.annotate(load_fixture(), counts)
        self.assertFalse(any(s["rare_here"] for s in a["species"]))
        self.assertEqual(a["headline"], verdict.annotate(load_fixture())["headline"])


class FormatTests(unittest.TestCase):
    def test_fmt_day(self):
        self.assertEqual(verdict.fmt_day("2026-10-10"), "Sat, Oct 10")
        self.assertEqual(verdict.fmt_range(["2026-10-10", "2026-10-11"]), "Sat, Oct 10 - Sun, Oct 11")

    def test_plural(self):
        self.assertEqual(verdict.plural("red maple"), "red maples")
        self.assertEqual(verdict.plural("American beech"), "American beeches")
        self.assertEqual(verdict.plural("sassafras"), "sassafras")
        self.assertEqual(verdict.join_names(["a", "b", "c"]), "a, b and c")


if __name__ == "__main__":
    unittest.main()
