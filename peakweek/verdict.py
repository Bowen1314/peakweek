"""Turn a forecast() result into a weekend verdict per species.

`annotate(result, nearby=None)` returns a deep copy of the forecast with a few fields added. It never changes
the model's probabilities. Everything is derived from the result's own `days` (never the clock),
so a saved forecast always reads the same.

The weekend
    The first Saturday in `days` plus the Sunday after it, if that Sunday is also in `days`.
    If `days` starts on a Sunday, the weekend is just that Sunday. A species' weekend numbers
    are the plain average of its daily probabilities over those one or two days.

The best day
    The day in `days` with the highest chance of fall color. Ties go to the earliest day.

The verdict (first rule that matches wins; every forecast gets exactly one)
    1. late      the weekend's chance of "no live leaves" is 35% or more.
    2. go        the weekend's chance of fall color is 50% or more.
    3. late      color is already fading: the best day is today or comes before the weekend,
                 and from the best day to the weekend the chance of color falls while the chance
                 of bare branches rises.
    4. starting  the weekend's chance of fall color is 25% or more.
    5. wait      anything else (mostly green); the text says when color is most likely.

The headline
    "Go see" names up to three species with a "go" verdict, most colorful first. If none is a
    "go", it names the species that are starting to turn, or the late ones that still show
    color. "Come back" names one more species (not already named) that is starting or waiting,
    whose best day is after the weekend and whose best chance of color is at least 25%: the one
    with the earliest best day (ties: the higher chance). Outside the training region the
    headline is prefixed with a warning.

Rarely recorded here
    `nearby` is {taxon_id: research-grade iNaturalist observations within 50 km} (peakweek.nearby),
    or None when that check could not be made. A species with fewer than 5 is "rare here": it
    keeps its verdict and numbers, its text starts with "Rarely recorded within 50 km;", the card
    lists it last, and the headline never names it (nor counts it for the summary sentences).
"""

from __future__ import annotations

import copy
import datetime as _dt
from typing import Dict, List, Optional, Sequence

KEYS = ("green", "colored", "bare")

LATE_BARE = 0.35       # weekend P(no live leaves) at or above this -> late
GO_COLORED = 0.50      # weekend P(colored) at or above this -> go
START_COLORED = 0.25   # weekend P(colored) at or above this -> starting
DROPPING_BARE = 0.25   # a "go" with at least this much bare gets "leaves are starting to fall"
BETTER_LATER = 0.10    # a "go" whose best day after the weekend is this much better gets a hint

VERDICTS = ("go", "starting", "wait", "late")

NEARBY_RADIUS_KM = 50
NEARBY_SOURCE = "iNaturalist research-grade observations"
RARE_BELOW = 5         # fewer research-grade observations than this within 50 km -> rare here
RARE_PREFIX = "Rarely recorded within %d km; " % NEARBY_RADIUS_KM

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def parse_day(day) -> _dt.date:
    if isinstance(day, _dt.date):
        return day
    return _dt.date.fromisoformat(str(day)[:10])


def fmt_day(day) -> str:
    """'2026-10-10' -> 'Sat, Oct 10' (locale independent)."""
    d = parse_day(day)
    return "%s, %s %d" % (_WEEKDAYS[d.weekday()], _MONTHS[d.month - 1], d.day)


def fmt_short(day) -> str:
    """'2026-10-10' -> 'Oct 10'."""
    d = parse_day(day)
    return "%s %d" % (_MONTHS[d.month - 1], d.day)


def fmt_range(days: Sequence) -> str:
    """Weekend label: 'Sat, Oct 10 - Sun, Oct 11' or a single day."""
    if not days:
        return ""
    if len(days) == 1:
        return fmt_day(days[0])
    return "%s - %s" % (fmt_day(days[0]), fmt_day(days[-1]))


def plural(name: str) -> str:
    """'red maple' -> 'red maples', 'American beech' -> 'American beeches', 'sassafras' unchanged."""
    if name.endswith("s"):
        return name
    if name.endswith(("ch", "sh", "x", "z")):
        return name + "es"
    return name + "s"


def join_names(names: Sequence[str]) -> str:
    names = list(names)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return "%s and %s" % (", ".join(names[:-1]), names[-1])


def weekend_indices(days: Sequence) -> List[int]:
    """Indices into `days` of the weekend to report on (see module docstring)."""
    dates = [parse_day(d) for d in days]
    for i, d in enumerate(dates):
        if d.weekday() == 5:  # Saturday
            idx = [i]
            if i + 1 < len(dates) and dates[i + 1] == d + _dt.timedelta(days=1):
                idx.append(i + 1)
            return idx
        if d.weekday() == 6:  # Sunday before any Saturday: the window starts on a Sunday
            return [i]
    raise ValueError("forecast days contain no Saturday or Sunday")


def best_index(daily: Sequence[dict]) -> int:
    """argmax P(colored); ties go to the earliest day."""
    best = 0
    for i in range(1, len(daily)):
        if daily[i]["colored"] > daily[best]["colored"]:
            best = i
    return best


def _mean(daily: Sequence[dict], idx: Sequence[int]) -> Dict[str, float]:
    return {k: sum(float(daily[i][k]) for i in idx) / len(idx) for k in KEYS}


def best_phrase(days: Sequence, daily: Sequence[dict], b: int, weekend: Sequence[int]) -> str:
    """'best this weekend' / 'best around Sat, Oct 17' / '... or later' if still rising at the end."""
    if b in weekend:
        return "best this weekend"
    text = "best around " + fmt_day(days[b])
    if b == len(days) - 1 and b > 0 and daily[b]["colored"] > daily[b - 1]["colored"]:
        text += " or later"
    return text


def classify(days: Sequence, daily: Sequence[dict], weekend: Sequence[int]):
    """Return (verdict, verdict_text, weekend_means, best_index) for one species."""
    wk = _mean(daily, weekend)
    b = best_index(daily)
    best_colored = float(daily[b]["colored"])
    peak_behind = (
        (b == 0 or b < weekend[0])
        and wk["colored"] < best_colored
        and wk["bare"] > float(daily[b]["bare"])
    )
    if wk["bare"] >= LATE_BARE:
        if wk["colored"] >= START_COLORED:
            return "late", "Many leaves are already down; go soon", wk, b
        return "late", "Most leaves are already down", wk, b
    if wk["colored"] >= GO_COLORED:
        text = "Go this weekend"
        if wk["bare"] >= DROPPING_BARE:
            text += "; leaves are starting to fall"
        elif b > weekend[-1] and best_colored - wk["colored"] >= BETTER_LATER:
            text += "; even more color around " + fmt_day(days[b])
        return "go", text, wk, b
    if peak_behind:
        return "late", "Color is fading; go soon", wk, b
    if wk["colored"] >= START_COLORED:
        return "starting", "Starting to turn; " + best_phrase(days, daily, b, weekend), wk, b
    if best_colored < START_COLORED:
        return "wait", "Mostly green through " + fmt_day(days[-1]), wk, b
    return "wait", "Mostly green; " + best_phrase(days, daily, b, weekend), wk, b


def _check(result: dict) -> None:
    days = result.get("days")
    if not days:
        raise ValueError("forecast has no days")
    for sp in result.get("species", []):
        daily = sp.get("daily", [])
        if len(daily) != len(days):
            raise ValueError("species %s has %d days, expected %d" % (sp.get("common"), len(daily), len(days)))
        for day, row in zip(days, daily):
            if row.get("date") is not None and str(row["date"])[:10] != str(day)[:10]:
                raise ValueError("species %s: date %s does not match %s" % (sp.get("common"), row["date"], day))


def make_headline(result: dict, weekend: Sequence[int]) -> str:
    days = result["days"]
    everything = result.get("species", [])
    species = [s for s in everything if not s.get("rare_here")]
    idx = {d: i for i, d in enumerate(days)}
    if everything and not species:
        sentence = "None of these trees is commonly recorded within %d km of here." % NEARBY_RADIUS_KM
        if result.get("in_region") is False:
            sentence = "Rough guide only (outside the area the model was trained on): " + sentence[0].lower() + sentence[1:]
        return sentence

    def by_weekend_color(group):
        return sorted(group, key=lambda s: (-s["weekend"]["colored"], s["common"]))

    go = by_weekend_color([s for s in species if s["verdict"] == "go"])[:3]
    starting = by_weekend_color([s for s in species if s["verdict"] == "starting"])[:3]
    late_colored = by_weekend_color(
        [s for s in species if s["verdict"] == "late" and s["weekend"]["colored"] >= START_COLORED])[:3]

    if go:
        named = go
        first = "This weekend, go see " + join_names(plural(s["common"]) for s in go)
    elif starting:
        named = starting
        first = "No tree here is in full color yet this weekend, but %s are starting to turn" % join_names(
            plural(s["common"]) for s in starting)
    elif late_colored:
        named = late_colored
        first = "Color is fading this weekend; %s still show some color" % join_names(
            plural(s["common"]) for s in late_colored)
    elif any(s["verdict"] == "late" for s in species):
        named = []
        first = "Most leaves are already down this weekend"
    else:
        named = []
        first = "Still mostly green this weekend"

    named_ids = {id(s) for s in named}
    candidates = [
        s for s in species
        if id(s) not in named_ids
        and s["verdict"] in ("starting", "wait")
        and idx[s["best_day"]] > weekend[-1]
        and s["best_colored"] >= START_COLORED
    ]
    candidates.sort(key=lambda s: (idx[s["best_day"]], -s["best_colored"], s["common"]))
    if candidates:
        nxt = candidates[0]
        sentence = "%s; come back around %s for %s." % (first, fmt_day(nxt["best_day"]), plural(nxt["common"]))
    elif not named and not any(s["verdict"] == "late" for s in species):
        sentence = "%s, and little color is expected through %s." % (first, fmt_day(days[-1]))
    else:
        sentence = first + "."
    if result.get("in_region") is False:
        # Every sentence above starts with a fixed word ("This", "No", ...), safe to lowercase.
        sentence = "Rough guide only (outside the area the model was trained on): " + sentence[0].lower() + sentence[1:]
    return sentence


def annotate(result: dict, nearby: Optional[Dict[int, int]] = None) -> dict:
    """Return a copy of `result` with weekend numbers, best day, verdict and a headline added.

    Per species: weekend {green, colored, bare}, best_day, best_colored, verdict, verdict_text, and
    with `nearby`: nearby_observations, rare_here. Top level: weekend_dates, headline, nearby
    {radius_km, source, available}. The daily probabilities are left exactly as they were.
    """
    _check(result)
    out = copy.deepcopy(result)
    days = [str(d)[:10] for d in out["days"]]
    weekend = weekend_indices(days)
    for sp in out.get("species", []):
        daily = sp["daily"]
        verdict, text, wk, b = classify(days, daily, weekend)
        sp["weekend"] = wk
        sp["best_day"] = days[b]
        sp["best_colored"] = float(daily[b]["colored"])
        sp["verdict"] = verdict
        sp["verdict_text"] = text
        sp.pop("nearby_observations", None)  # never keep marks from an earlier annotate()
        sp.pop("rare_here", None)
        if nearby is not None and sp.get("taxon_id") is not None:
            tid = int(sp["taxon_id"])
            n = int(nearby.get(tid, nearby.get(str(tid), 0)))  # int keys, or str keys from JSON
            sp["nearby_observations"] = n
            sp["rare_here"] = n < RARE_BELOW
            if sp["rare_here"]:
                # every verdict text starts with a fixed word ("Go", "Most", ...), safe to lowercase
                sp["verdict_text"] = RARE_PREFIX + text[0].lower() + text[1:]
    out["nearby"] = {"radius_km": NEARBY_RADIUS_KM, "source": NEARBY_SOURCE, "available": nearby is not None}
    out["weekend_dates"] = [days[i] for i in weekend]
    out["headline"] = make_headline(dict(out, days=days), weekend)
    return out
