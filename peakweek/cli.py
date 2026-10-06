"""Command line: print the peakweek card for a place.

    python -m peakweek "New Brunswick, NJ"
    python -m peakweek --lat 40.49 --lon -74.45
    python -m peakweek --fixture examples/forecast_fixture.json   (no TabPFN needed)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from typing import Callable, List, Optional

from . import card, geocode, nearby, verdict


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m peakweek",
        description="Which trees near you are turning color this weekend, and which day to go out.")
    p.add_argument("place", nargs="?", help='place name like "New Brunswick, NJ", or "lat, lon"')
    p.add_argument("--lat", type=float, help="latitude (use with --lon)")
    p.add_argument("--lon", type=float, help="longitude (use with --lat)")
    p.add_argument("--fixture", metavar="PATH", help="render a saved forecast() result instead of running the model")
    p.add_argument("--today", metavar="YYYY-MM-DD", help="forecast as if today were this date (passed to the model)")
    p.add_argument("--json", action="store_true", help="print the annotated result as JSON instead of the card")
    return p


def main(argv: Optional[List[str]] = None, nearby_fn: Optional[Callable] = nearby.counts) -> int:
    """`nearby_fn(lat, lon)` is the iNaturalist local check (injectable for tests; None skips it)."""
    args = build_parser().parse_args(argv)
    try:
        if args.fixture:
            with open(args.fixture, encoding="utf-8") as f:
                result = json.load(f)
            place = result.get("place") or {}
            lat, lon = place.get("lat"), place.get("lon")
        else:
            lat, lon, name = resolve_place(args)
            today = _dt.date.fromisoformat(args.today) if args.today else None
            print("Forecasting for %s (%.4f, %.4f); this takes about a minute..." % (name, lat, lon), file=sys.stderr)
            from .forecast import forecast  # imported lazily: needs TabPFN, which fixture mode does not
            result = forecast(lat, lon, place_name=name, today=today)
        annotated = verdict.annotate(result, local_check(nearby_fn, lat, lon))
    except ImportError as e:
        print("peakweek: the live forecast needs the model's dependencies (requirements.txt): %s\n"
              "Use --fixture PATH to render a saved forecast instead." % e, file=sys.stderr)
        return 2
    except (OSError, ValueError, LookupError) as e:
        print("peakweek: %s" % e, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(annotated, indent=1))
    else:
        sys.stdout.write(card.render_text(annotated))
    return 0


def local_check(nearby_fn: Optional[Callable], lat, lon):
    """{taxon_id: count} from the iNaturalist check, or None (unavailable) on any failure."""
    if nearby_fn is None or lat is None or lon is None:
        return None
    try:
        return nearby_fn(float(lat), float(lon))
    except Exception:  # optional extra: never let it stop the card
        return None


def resolve_place(args):
    if args.lat is not None or args.lon is not None:
        if args.lat is None or args.lon is None:
            raise ValueError("give both --lat and --lon")
        geocode.check_latlon(args.lat, args.lon)
        return args.lat, args.lon, args.place
    if not args.place:
        raise ValueError('give a place ("New Brunswick, NJ"), --lat/--lon, or --fixture')
    ll = geocode.parse_latlon(args.place)
    if ll is not None:
        return ll[0], ll[1], None
    found = geocode.search(args.place, count=5)
    if not found:
        raise LookupError("no place found for %r" % args.place)
    top = found[0]
    print("Place: %s" % top["label"], file=sys.stderr)
    return top["lat"], top["lon"], top["short"]
