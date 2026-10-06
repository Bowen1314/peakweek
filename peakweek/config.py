"""Shared constants for the peakweek core (data, features, model, forecast)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = Path(os.environ.get("PEAKWEEK_CACHE_DIR", ROOT / ".cache"))

USER_AGENT = "peakweek/0.1 (hackathon project; contact via GitHub)"

# Training region: US Northeast and Mid-Atlantic.
REGION = {"lat_min": 38.5, "lat_max": 47.5, "lon_min": -80.5, "lon_max": -66.9}
REGION_NOTE = "Trained on the US Northeast and Mid-Atlantic (lat 38.5-47.5, lon -80.5 to -66.9)."

# Daily weather is aggregated in local time for both training (ERA5 archive) and the app (forecast API).
TIMEZONE = "America/New_York"

# Every seasonal accumulator starts on this month/day of the observation's year.
SEASON_START = (9, 1)

# Weather grid: observations are binned to cells of this size (degrees). Weather for a cell is fetched at the
# cell's "weather point": the mean coordinate of the cell's training observations (so coastal cells are not
# sampled over the ocean); cells without training observations fall back to the geometric center.
# 1.0 deg keeps the full 2018-2025 ERA5 pull at ~5,400 weighted Open-Meteo calls (0.5 deg would need ~15,000).
GRID_DEG = 1.0

LABELS = ("green", "colored", "bare")
LABEL_TO_INT = {name: i for i, name in enumerate(LABELS)}

# iNaturalist controlled term 36 = "Leaves"; values: 37 Breaking Leaf Buds (ignored), 38 Green, 39 Colored, 40 No Live.
INAT_LEAVES_TERM = 36
INAT_VALUE_TO_LABEL = {38: "green", 39: "colored", 40: "bare"}
INAT_IGNORED_VALUES = {37}

TRAIN_YEARS = tuple(range(2018, 2026))


def in_region(lat: float, lon: float) -> bool:
    """True if (lat, lon) is inside the training box (inclusive bounds)."""
    return (
        REGION["lat_min"] <= lat <= REGION["lat_max"]
        and REGION["lon_min"] <= lon <= REGION["lon_max"]
    )
