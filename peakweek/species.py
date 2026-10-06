"""The fall-color tree species peakweek forecasts (iNaturalist taxon ids)."""

from __future__ import annotations

SPECIES = [
    {"taxon_id": 48098, "common": "red maple", "scientific": "Acer rubrum"},
    {"taxon_id": 52543, "common": "sugar maple", "scientific": "Acer saccharum"},
    {"taxon_id": 49658, "common": "sweetgum", "scientific": "Liquidambar styraciflua"},
    {"taxon_id": 49005, "common": "northern red oak", "scientific": "Quercus rubra"},
    {"taxon_id": 49202, "common": "American beech", "scientific": "Fagus grandifolia"},
    {"taxon_id": 54802, "common": "black gum", "scientific": "Nyssa sylvatica"},
    {"taxon_id": 54795, "common": "sassafras", "scientific": "Sassafras albidum"},
    {"taxon_id": 54763, "common": "Norway maple", "scientific": "Acer platanoides"},
]

TAXON_IDS = [s["taxon_id"] for s in SPECIES]
BY_TAXON = {s["taxon_id"]: s for s in SPECIES}
# Stable integer code per species (used as a categorical feature).
SPECIES_CODE = {s["taxon_id"]: i for i, s in enumerate(SPECIES)}
