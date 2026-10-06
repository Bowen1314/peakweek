import copy
import json
import re
import unittest
from pathlib import Path

from peakweek import card, fieldnotes, verdict

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "examples" / "forecast_fixture.json"
KNOWN_IDS = (48098, 52543, 49658, 49005, 49202, 54802, 54795, 54763)


def fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def annotated(mutate=None):
    raw = fixture()
    if mutate:
        mutate(raw)
    return verdict.annotate(raw)


class WordingTests(unittest.TestCase):
    def test_in_ten(self):
        self.assertEqual(card.in_ten(0.68), "about 7 in 10")
        self.assertEqual(card.in_ten(0.02), "fewer than 1 in 10")
        self.assertEqual(card.in_ten(0.97), "nearly all")
        self.assertEqual(card.in_ten(0.55, tenths=5), "about 5 in 10")
        self.assertEqual(card.in_ten(0.07, tenths=0), "fewer than 1 in 10")

    def test_whole_parts_always_add_up(self):
        for c in range(0, 101, 7):
            for b in range(0, 101 - c, 3):
                probs = {"green": (100 - c - b) / 100, "colored": c / 100, "bare": b / 100}
                self.assertEqual(sum(card.whole_parts(probs).values()), 10, probs)
                self.assertEqual(sum(card.whole_parts(probs, 20).values()), 20, probs)

    def test_chance_sentence_is_per_tree(self):
        a = annotated()
        red = next(s for s in a["species"] if s["common"] == "red maple")
        self.assertEqual(card.chance_sentence(red),
                         "About 6 in 10 red maples you find this weekend should show fall color.")


class TextCardTests(unittest.TestCase):
    def test_fits_72_columns(self):
        for a in (annotated(), annotated(lambda r: r.update(in_region=False))):
            text = card.render_text(a)
            self.assertLessEqual(max(len(line) for line in text.splitlines()), 72)

    def test_long_names_still_fit(self):
        def longer(r):
            r["place"]["name"] = "A Very Long Place Name, Somewhere Along The Delaware Water Gap, PA"
            r["species"][0]["common"] = "extraordinarily long common name of a tree"
        text = card.render_text(annotated(longer))
        self.assertLessEqual(max(len(line) for line in text.splitlines()), 72)

    def test_contents(self):
        text = card.render_text(annotated())
        self.assertIn(card.CREDIT, " ".join(text.split()))  # the text card wraps at 72 columns
        self.assertIn("Synthetic example", text)
        self.assertIn("Weekend of Sat, Oct 10 - Sun, Oct 11", text)
        self.assertIn("About 6 in 10 red maples you find this weekend should show", text)
        self.assertIn("https://www.inaturalist.org/", text)
        self.assertIn("1,000 past observations", text)
        self.assertNotIn("%", text)  # chances are "N in 10", never a percentage of the canopy

    def test_bar_matches_numbers(self):
        text = card.render_text(annotated())
        for m in re.finditer(r"\[([-#.]{20})\]  (\d+) green, (\d+) color, (\d+) bare", text):
            bar, g, c, b = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            self.assertEqual((bar.count("-"), bar.count("#"), bar.count(".")), (2 * g, 2 * c, 2 * b))

    def test_out_of_region_note_shown(self):
        text = card.render_text(annotated(lambda r: r.update(in_region=False)))
        self.assertIn("outside the area the model was trained on", " ".join(text.split()))
        self.assertIn("lat 38.5-47.5", " ".join(text.split()))

    def test_no_synthetic_banner_for_real_results(self):
        text = card.render_text(annotated(lambda r: r.pop("synthetic")))
        self.assertNotIn("Synthetic example", text)


class NearbyCardTests(unittest.TestCase):
    """Species rarely recorded within 50 km are listed last and muted; a failed check says so."""

    def counts(self, **override):
        c = {t: 300 for t in KNOWN_IDS}
        c.update({int(k[1:]): v for k, v in override.items()})
        return c

    def test_rare_listed_last_in_both_renderings(self):
        raw = fixture()
        first = card.display_order(verdict.annotate(raw)["species"])[0]  # most colorful
        a = verdict.annotate(raw, self.counts(**{"t%d" % first["taxon_id"]: 2}))
        order = card.display_order(a["species"])
        self.assertEqual(order[-1]["common"], first["common"])
        self.assertTrue(order[-1]["rare_here"])
        self.assertFalse(any(s["rare_here"] for s in order[:-1]))
        h = card.render_html(a)
        self.assertEqual(h.count(" rare\">"), 1)
        last_li = h.rindex('<li class="sp ')
        self.assertTrue(h.startswith('<li class="sp v-%s rare">' % order[-1]["verdict"], last_li))
        self.assertIn("Rarely recorded within 50 km; ", h[last_li:])
        self.assertIn(card.RARE_LINE, h)
        self.assertNotIn(card.NEARBY_UNAVAILABLE_LINE, h)
        t = card.render_text(a)
        flat = " ".join(t.split())
        self.assertIn(card.RARE_LINE, flat)
        names = [s["common"] for s in order]
        positions = [t.index(" %s (" % n) for n in names]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn(first["common"], a["headline"])

    def test_all_common_no_note(self):
        a = verdict.annotate(fixture(), self.counts())
        self.assertNotIn(" rare\"", card.render_html(a))
        for out in (card.render_html(a), " ".join(card.render_text(a).split())):
            self.assertNotIn(card.RARE_LINE, out)
            self.assertNotIn(card.NEARBY_UNAVAILABLE_LINE, out)

    def test_unavailable_line(self):
        a = verdict.annotate(fixture(), None)
        self.assertIn('<p class="nearby">%s</p>' % card.NEARBY_UNAVAILABLE_LINE, card.render_html(a))
        self.assertIn(card.NEARBY_UNAVAILABLE_LINE, " ".join(card.render_text(a).split()))
        self.assertTrue(all(len(line) <= 72 for line in card.render_text(a).splitlines()))

    def test_old_results_without_nearby_say_nothing(self):
        a = verdict.annotate(fixture())
        del a["nearby"]
        self.assertNotIn('class="nearby"', card.render_html(a))
        self.assertNotIn("Local check", card.render_text(a))


class ReplayLineTests(unittest.TestCase):
    def test_wording(self):
        self.assertEqual(card.replay_line({}), "")
        same = {"replay": {"place": "New Brunswick, NJ", "made": "2026-10-06", "searched": None}}
        self.assertEqual(card.replay_line(same), "Replaying a saved forecast for New Brunswick, NJ (made Oct 6). "
                                                 "Run without --fixture for a live forecast.")
        other = {"replay": {"place": "New Brunswick, NJ", "made": "2026-10-06", "searched": "Burlington, VT"}}
        self.assertEqual(card.replay_line(other), "You searched Burlington, VT; this is the saved forecast for "
                                                  "New Brunswick, NJ (made Oct 6). Run without --fixture for a live forecast.")

    def test_shown_in_both_renderings_only_when_present(self):
        a = annotated()
        self.assertNotIn("banner-replay", card.render_html(a))
        self.assertNotIn("saved forecast", card.render_text(a))
        a["replay"] = {"place": "New Brunswick, NJ", "made": "2026-10-06", "searched": "Ithaca, NY"}
        self.assertIn('class="banner banner-replay"', card.render_html(a))
        text = card.render_text(a)
        self.assertIn("You searched Ithaca, NY; this is the saved forecast", " ".join(text.split()))
        self.assertTrue(all(len(line) <= 72 for line in text.splitlines()))


class HtmlCardTests(unittest.TestCase):
    def test_structure(self):
        h = card.render_html(annotated())
        self.assertTrue(h.startswith('<article class="card is-synthetic">'))
        self.assertEqual(h.count('<li class="sp '), 8)
        self.assertEqual(h.count('<li class="day'), 8 * 15)
        self.assertIn("banner-synthetic", h)
        self.assertIn(card.CREDIT, h)
        self.assertIn("chip-go", h)
        self.assertIn("Sat, Oct 10 – Sun, Oct 11", h)

    def test_most_colorful_first(self):
        h = card.render_html(annotated())
        self.assertLess(h.index("red maple"), h.index("Norway maple"))

    def test_everything_is_escaped(self):
        def evil(r):
            r["place"]["name"] = "<script>alert(1)</script>"
            r["species"][0]["common"] = "<img src=x onerror=alert(1)>"
        h = card.render_html(annotated(evil))
        self.assertNotIn("<script>", h)
        self.assertNotIn("<img", h)
        self.assertIn("&lt;script&gt;", h)

    def test_region_banner(self):
        h = card.render_html(annotated(lambda r: r.update(in_region=False)))
        self.assertIn("banner-region", h)
        self.assertIn("lat 38.5-47.5", h)
        self.assertNotIn("banner-region", card.render_html(annotated()))

    def test_look_for_line_and_unknown_taxon(self):
        def unknown(r):
            r["species"][0]["taxon_id"] = 999999999
        h = card.render_html(annotated(unknown))
        self.assertEqual(h.count('class="look"'), 7)
        self.assertIn("Star-shaped leaves", h)


class FieldnoteTests(unittest.TestCase):
    def test_every_species_has_a_note(self):
        for tid in KNOWN_IDS:
            note = fieldnotes.look_for(tid)
            self.assertTrue(note and len(note) < 140, tid)

    def test_core_species_list_is_covered(self):
        try:
            from peakweek.species import SPECIES
        except ImportError:
            self.skipTest("peakweek.species not available")
        for s in SPECIES:
            self.assertIsNotNone(fieldnotes.look_for(s["taxon_id"]), s["common"])

    def test_unknown_or_bad_ids(self):
        self.assertIsNone(fieldnotes.look_for(1))
        self.assertIsNone(fieldnotes.look_for(None))
        self.assertIsNone(fieldnotes.look_for("x"))
        self.assertEqual(fieldnotes.look_for("49658"), fieldnotes.look_for(49658))


if __name__ == "__main__":
    unittest.main()
