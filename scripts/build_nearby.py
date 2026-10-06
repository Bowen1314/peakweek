#!/usr/bin/env python3
"""One-time (development) iNaturalist check for the daily site: data/nearby_cells.json.

For every published grid cell (peakweek/site.py), the research-grade iNaturalist observations of the
8 trees within 50 km of the cell's weather point (peakweek.nearby, one keyless call per cell, about one
call per second). The daily build only reads this file; it never calls iNaturalist.

    python3 scripts/build_nearby.py            # writes data/nearby_cells.json
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from peakweek import nearby, site  # noqa: E402


def main() -> int:
    cells = site.published_cells()
    out = {}
    for i, c in enumerate(cells):
        counts = None
        for attempt in range(4):
            counts = nearby.counts(c["lat"], c["lon"], cache_dir=None, timeout=30.0)
            if counts is not None:
                break
            time.sleep(10 * (attempt + 1))
        if counts is None:
            print("iNaturalist did not answer for %s; nothing written" % c["id"], file=sys.stderr)
            return 1
        out[c["id"]] = {"lat": c["lat"], "lon": c["lon"], "counts": {str(k): int(v) for k, v in counts.items()}}
        print("%3d/%d %s %s" % (i + 1, len(cells), c["id"], sum(counts.values())), flush=True)
        time.sleep(1.1)
    data = {
        "source": nearby.SOURCE, "radius_km": nearby.RADIUS_KM,
        "fetched_on": dt.date.today().isoformat(),
        "point": "each cell's weather point (mean coordinate of its training observations), rounded to 0.01 deg",
        "api": nearby.API_URL,
        "cells": out,
    }
    site.NEARBY_FILE.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    print("wrote %s (%d cells)" % (site.NEARBY_FILE, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
