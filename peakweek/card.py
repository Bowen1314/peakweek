"""Render an annotated forecast (see verdict.annotate) as a terminal card or an HTML fragment.

Wording rule: P(colored) is the chance that one tree of that species you come across shows
fall color ("about 7 in 10 red maples you find"), never a share of the canopy.
"""

from __future__ import annotations

import html
import math
import re
import textwrap
from typing import Dict, List, Optional, Sequence

from . import fieldnotes, verdict

CREDIT = ("Forecast: TabPFN v2 on iNaturalist observations + Open-Meteo weather. "
          "Built with PriorLabs-TabPFN. Weather data by Open-Meteo.com.")
CHANCE_NOTE = "Each chance is for one tree you come across, not the share of the canopy."
INAT_URL = "https://www.inaturalist.org/"
INAT_LINE = 'After your walk, add what you saw to iNaturalist with the "Leaves" annotation.'
SYNTHETIC_LINE = "Synthetic example: hand-made numbers for testing the page, not a model forecast."
OUTSIDE_LINE = "This place is outside the area the model was trained on, so treat the forecast loosely."
NEARBY_UNAVAILABLE_LINE = ("Local check unavailable: iNaturalist could not be reached to see which of "
                           "these trees are recorded within %d km, so all are listed as usual."
                           % verdict.NEARBY_RADIUS_KM)
REPLAY_TAIL = "Run without --fixture for a live forecast."
RARE_LINE = ("Listed last: trees with fewer than %d research-grade iNaturalist observations within %d km."
             % (verdict.RARE_BELOW, verdict.NEARBY_RADIUS_KM))

VERDICT_LABEL = {"go": "Go", "starting": "Starting", "wait": "Wait", "late": "Late"}
KEYS = verdict.KEYS


# ---------------------------------------------------------------- numbers and wording

def whole_parts(probs: Dict[str, float], total: int = 10) -> Dict[str, int]:
    """Split probabilities into whole parts of `total` that add up exactly (largest remainder)."""
    raw = {k: max(0.0, float(probs[k])) * total for k in KEYS}
    base = {k: int(math.floor(raw[k])) for k in KEYS}
    left = max(0, total - sum(base.values()))
    order = sorted(KEYS, key=lambda k: (-(raw[k] - base[k]), KEYS.index(k)))
    for k in order[:left]:
        base[k] += 1
    return base


def in_ten(p: float, tenths: Optional[int] = None) -> str:
    """0.68 -> 'about 7 in 10'; 0 tenths -> 'fewer than 1 in 10'; 10 tenths -> 'nearly all'.

    Pass `tenths` (from whole_parts) so the words agree with the numbers shown next to them.
    """
    n = tenths if tenths is not None else int(math.floor(float(p) * 10 + 0.5))
    if n <= 0:
        return "fewer than 1 in 10"
    if n >= 10:
        return "nearly all"
    return "about %d in 10" % n


def chance_sentence(sp: dict) -> str:
    """'About 6 in 10 red maples you find this weekend should show fall color.'"""
    parts = whole_parts(sp["weekend"])
    phrase = in_ten(sp["weekend"]["colored"], parts["colored"])
    return "%s %s you find this weekend should show fall color." % (
        phrase[0].upper() + phrase[1:], verdict.plural(sp["common"]))


def when_sentence(sp: dict, today: Optional[str]) -> str:
    """Verdict text plus the best day, unless the verdict text already names that day."""
    vt = sp["verdict_text"]
    best = verdict.fmt_day(sp["best_day"])
    if best in vt:
        return vt + "."
    day = ("today (%s)" % best) if today and str(today)[:10] == sp["best_day"] else best
    return "%s. Best day: %s, %s." % (vt, day, in_ten(sp["best_colored"]))


def display_order(species: Sequence[dict]) -> List[dict]:
    """Most colorful this weekend first; species rarely recorded nearby after all the others."""
    return sorted(species, key=lambda s: (bool(s.get("rare_here")), -s["weekend"]["colored"], s["common"]))


def replay_line(result: dict) -> str:
    """Server fixture mode with a saved real forecast: say it is a replay (and of which place)."""
    rp = result.get("replay")
    if not isinstance(rp, dict):
        return ""
    place = str(rp.get("place") or "another place")
    made = (" (made %s)" % verdict.fmt_short(rp["made"])) if rp.get("made") else ""
    if rp.get("searched"):
        first = "You searched %s; this is the saved forecast for %s%s." % (rp["searched"], place, made)
    else:
        first = "Replaying a saved forecast for %s%s." % (place, made)
    return first + " " + REPLAY_TAIL


def nearby_lines(result: dict) -> List[str]:
    """The local-check note for the card foot: unavailable, or what "listed last" means."""
    info = result.get("nearby")
    if not isinstance(info, dict):
        return []  # annotated before the local check existed: say nothing
    if info.get("available") is False:
        return [NEARBY_UNAVAILABLE_LINE]
    if any(s.get("rare_here") for s in result.get("species", [])):
        return [RARE_LINE]
    return []


def model_details(result: dict) -> str:
    """'TabPFN v2 classifier (open weights, CPU); 1,000 past observations as context; ...'"""
    model = result.get("model") or {}
    weather = result.get("weather") or {}
    bits = []
    if model.get("name"):
        bits.append(str(model["name"]))
    if model.get("context_rows"):
        bits.append("{:,} past observations as context".format(int(model["context_rows"])))
    if weather.get("archive_from"):
        bits.append("weather since %s plus a 2-week forecast" % verdict.fmt_short(weather["archive_from"]))
    elif weather.get("past_days"):
        bits.append("%d days of weather" % int(weather["past_days"]))
    if model.get("seconds") is not None:
        bits.append("ran in %.0f s" % float(model["seconds"]))
    gen = fmt_generated(result.get("generated_at"))
    if gen:
        bits.append("made " + gen)
    return "; ".join(bits)


def fmt_generated(stamp) -> str:
    """'2026-10-06T14:00:00Z' -> 'Oct 6, 14:00 UTC'."""
    if not stamp:
        return ""
    s = str(stamp)
    try:
        day = verdict.fmt_short(s[:10])
    except ValueError:
        return s
    if len(s) >= 16 and s[10] == "T":
        zone = " UTC" if s.endswith("Z") or s.endswith("+00:00") else ""
        return "%s, %s%s" % (day, s[11:16], zone)
    return day


def _tenth(p: float) -> int:
    return min(9, int(math.floor(float(p) * 10 + 0.5)))


def place_name(result: dict) -> str:
    place = result.get("place") or {}
    if place.get("name"):
        return str(place["name"])
    if place.get("lat") is not None and place.get("lon") is not None:
        return "%.2f, %.2f" % (float(place["lat"]), float(place["lon"]))
    return "your area"


# ---------------------------------------------------------------- plain text

def _bar_text(wk: Dict[str, float]) -> str:
    """20-character bar, two characters per tenth, so it matches the "(in 10)" numbers."""
    parts = whole_parts(wk)
    return "[" + "--" * parts["green"] + "##" * parts["colored"] + ".." * parts["bare"] + "]"


_KEEP_TOGETHER = re.compile(
    r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}\b"
    r"|\b\d+ in 10\b")


def _nbsp(text: str) -> str:
    """Glue dates and 'N in 10' together so textwrap does not split them."""
    return _KEEP_TOGETHER.sub(lambda m: m.group(0).replace(" ", "\u00a0"), text)


def render_text(result: dict, width: int = 72) -> str:
    """The card for a terminal: plain ASCII, no line longer than `width` (default 72)."""
    days = result["days"]
    weekend = set(result.get("weekend_dates") or [])
    out: List[str] = []

    def para(text: str, indent: str = "") -> None:
        lines = textwrap.wrap(_nbsp(text), width=width, initial_indent=indent, subsequent_indent=indent,
                              break_on_hyphens=False) or [indent.rstrip()]
        out.extend(line.replace("\u00a0", " ") for line in lines)

    para("peakweek: " + place_name(result))
    made = (" (forecast made %s)" % verdict.fmt_day(result["today"])) if result.get("today") else ""
    para("Weekend of " + verdict.fmt_range(result.get("weekend_dates") or []) + made)
    if result.get("synthetic"):
        para("** " + SYNTHETIC_LINE + " **")
    if replay_line(result):
        para("** " + replay_line(result) + " **")
    if result.get("in_region") is False:
        para("** " + OUTSIDE_LINE + " " + str(result.get("region_note") or "") + " **")
    out.append("")
    para(result.get("headline", ""))
    out.append("")
    para("Bars show this weekend:  - green   # fall color   . no live leaves")
    out.append("")

    pad = " " * 10
    span = "%s-%s" % (verdict.fmt_short(days[0]), verdict.fmt_short(days[-1]))
    for sp in display_order(result.get("species", [])):
        label = VERDICT_LABEL.get(sp["verdict"], sp["verdict"]).upper()
        para("%-10s%s (%s)" % (label, sp["common"], sp.get("scientific", "")))
        parts = whole_parts(sp["weekend"])
        out.append(pad + _bar_text(sp["weekend"]) + "  %d green, %d color, %d bare (in 10)" % (
            parts["green"], parts["colored"], parts["bare"]))
        para(chance_sentence(sp) + " " + when_sentence(sp, result.get("today")), pad)
        note = fieldnotes.look_for(sp.get("taxon_id"))
        if note:
            para("Look for: " + note, pad)
        # one digit per day (chance of color, in tenths), weekend days in brackets
        strip = ""
        for i, (d, row) in enumerate(zip(days, sp["daily"])):
            opens = d in weekend and (i == 0 or days[i - 1] not in weekend)
            closes = d in weekend and (i == len(days) - 1 or days[i + 1] not in weekend)
            strip += ("[" if opens else (" " if i else "")) + str(_tenth(row["colored"])) + ("]" if closes else "")
        out.append(pad + "Color %s (x in 10): %s" % (span, strip))
        out.append("")

    for line in nearby_lines(result):
        para(line)
    para(CREDIT)
    details = model_details(result)
    if details:
        para(details + ".")
    para(CHANCE_NOTE)
    para(INAT_LINE + " " + INAT_URL)
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------- HTML fragment

def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _pct(p: float) -> str:
    return "%.1f%%" % (max(0.0, min(1.0, float(p))) * 100)


def _strip_html(sp: dict, days: Sequence[str], weekend: set) -> str:
    cells = []
    best = sp["best_day"]
    for d, row in zip(days, sp["daily"]):
        cls = ["day"]
        if d in weekend:
            cls.append("wk")
        if d == best:
            cls.append("best")
        dd = verdict.parse_day(d)
        title = "%s: %s" % (verdict.fmt_day(d), in_ten(row["colored"]))
        cells.append('<li class="%s" title="%s"><i style="height:%s"></i><b>%s</b></li>' % (
            " ".join(cls), _e(title), _pct(row["colored"]), "MTWTFSS"[dd.weekday()]))
    label = "Chance of fall color by day, %s to %s, in tenths: %s" % (
        verdict.fmt_day(days[0]), verdict.fmt_day(days[-1]),
        ", ".join(str(_tenth(r["colored"])) for r in sp["daily"]))
    return (
        '<figure class="days"><ol class="strip" role="img" aria-label="%s">%s</ol>'
        '<figcaption><span>%s</span><span>chance of color by day</span><span>%s</span></figcaption></figure>'
        % (_e(label), "".join(cells), _e(verdict.fmt_short(days[0])), _e(verdict.fmt_short(days[-1]))))


def _species_html(sp: dict, days: Sequence[str], weekend: set, today: Optional[str]) -> str:
    wk = sp["weekend"]
    parts = whole_parts(wk)
    v = sp["verdict"]
    note = fieldnotes.look_for(sp.get("taxon_id"))
    bar_label = "This weekend: about %d in 10 green, %d in 10 with fall color, %d in 10 with no live leaves" % (
        parts["green"], parts["colored"], parts["bare"])
    rows = [
        '<li class="sp v-%s%s">' % (_e(v), " rare" if sp.get("rare_here") else ""),
        '<div class="sp-head"><span class="chip chip-%s">%s</span>'
        '<h3><span class="common">%s</span> <i class="sci">%s</i></h3></div>' % (
            _e(v), _e(VERDICT_LABEL.get(v, v)), _e(sp["common"]), _e(sp.get("scientific", ""))),
        '<div class="bar" role="img" aria-label="%s"><span class="g" style="width:%s"></span>'
        '<span class="c" style="width:%s"></span><span class="b" style="width:%s"></span></div>' % (
            _e(bar_label), _pct(wk["green"]), _pct(wk["colored"]), _pct(wk["bare"])),
        '<p class="nums">%d green &middot; <strong>%d color</strong> &middot; %d bare <span>(in 10)</span></p>' % (
            parts["green"], parts["colored"], parts["bare"]),
        '<p class="chance">%s</p>' % _e(chance_sentence(sp)),
        '<p class="when">%s</p>' % _e(when_sentence(sp, today)),
    ]
    if note:
        rows.append('<p class="look"><span>Look for</span> %s</p>' % _e(note))
    rows.append(_strip_html(sp, days, weekend))
    rows.append("</li>")
    return "".join(rows)


def render_html(result: dict) -> str:
    """The card as an HTML fragment (an <article>); every value is escaped."""
    days = [str(d)[:10] for d in result["days"]]
    weekend = set(result.get("weekend_dates") or [])
    head = []
    if result.get("synthetic"):
        head.append('<p class="banner banner-synthetic" role="note">%s</p>' % _e(SYNTHETIC_LINE))
    if replay_line(result):
        head.append('<p class="banner banner-replay" role="note">%s</p>' % _e(replay_line(result)))
    if result.get("in_region") is False:
        head.append('<p class="banner banner-region" role="note">%s %s</p>' % (
            _e(OUTSIDE_LINE), _e(result.get("region_note") or "")))
    made = ""
    if result.get("today"):
        made = ' <span class="made">Forecast made %s</span>' % _e(verdict.fmt_day(result["today"]))
    head.append(
        '<header class="card-head"><p class="place">%s</p>'
        '<p class="weekend">Weekend of <strong>%s</strong>%s</p>'
        '<h2 class="headline">%s</h2>'
        '<p class="legend"><span class="key g"></span>green <span class="key c"></span>fall color '
        '<span class="key b"></span>no live leaves <span class="legend-note">(bars: this weekend)</span></p>'
        '</header>' % (
            _e(place_name(result)), _e(verdict.fmt_range(result.get("weekend_dates") or []).replace(" - ", " \u2013 ")), made,
            _e(result.get("headline", ""))))
    items = "".join(_species_html(sp, days, weekend, result.get("today")) for sp in display_order(result.get("species", [])))
    details = model_details(result)
    foot = (
        '<footer class="card-foot">%s<p class="credit">%s</p>%s<p class="note">%s</p></footer>' % (
            "".join('<p class="nearby">%s</p>' % _e(line) for line in nearby_lines(result)),
            _e(CREDIT), ('<p class="details">%s.</p>' % _e(details)) if details else "", _e(CHANCE_NOTE)))
    return '<article class="card%s">%s<ol class="species">%s</ol>%s</article>' % (
        " is-synthetic" if result.get("synthetic") else "", "".join(head), items, foot)
