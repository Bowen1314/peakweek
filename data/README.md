# peakweek data

All files here are produced by `python scripts/build_dataset.py` (raw API responses are cached under
`.cache/`, which is not committed). Pulled on **2026-10-06**.

| file | rows | what |
|---|---:|---|
| `observations.csv` | 6650 | one row per labeled iNaturalist observation (obs_id, observed_on, lat, lon rounded to 3 decimals, taxon_id, label, quality_grade, captive, license_code) |
| `features.csv` | 6650 | model-ready rows (same observations) with the features below |
| `climatology.csv` | 9476 | per weather cell and season day: mean over 2018-2025 of the season-to-date mean temperature (used by the app for the anomaly feature) |
| `build_summary.json` | | counts, drop reasons, Open-Meteo usage |

## iNaturalist query (API v1, keyless)

- `GET https://api.inaturalist.org/v1/observations` with `taxon_id=<species>`, `term_id=36` (Leaves, any value),
  `month=9,10,11`, `d1=2018-01-01`, `d2=2025-12-31`, bounding box `swlat=38.5 swlng=-80.5 nelat=47.5 nelng=-66.9`,
  `per_page=200`, paginated with `order_by=id&order=asc&id_above=...`, <= 1 request/second,
  User-Agent `peakweek/0.1 (hackathon project; contact via GitHub)`.
- Species (queried taxon id is stored, descendants included): red maple 48098, sugar maple 52543, sweetgum 49658,
  northern red oak 49005, American beech 49202, black gum 54802, sassafras 54795, Norway maple 54763.
- Label: Leaves annotations (attribute 36) with `vote_score >= 0`; 38 Green Leaves -> `green`, 39 Colored Leaves ->
  `colored`, 40 No Live Leaves -> `bare`; 37 Breaking Leaf Buds ignored. Observations whose counted annotations
  disagree are dropped (every one of these was Green + Colored on the same observation, i.e. probably a partly
  turned tree).
- Kept: quality grade research or needs_id (casual and captive/cultivated dropped); coordinates not obscured
  (`obscured`, `geoprivacy`, `taxon_geoprivacy`), positional accuracy <= 1000 m (or unknown), inside the box;
  non-null `license_code` (observations without a license are not redistributed here).

### Drop counts (all 8 species, Sep-Nov 2018-2025)

| step | observations |
|---|---:|
| returned by the API (any Leaves annotation) | 10877 |
| captive / cultivated | 879 |
| casual quality grade | 52 |
| obscured coordinates | 244 |
| positional accuracy > 1000 m | 455 |
| conflicting Leaves values (all Green + Colored) | 856 |
| no usable Leaves value (only Breaking Leaf Buds / downvoted) | 1 |
| **labeled** | 8390 |
| null license_code (not redistributed) | 1740 |
| **kept** (`observations.csv`) | 6650 |


## Weather and elevation

- Observations are binned to a 1.0 degree grid; each cell's weather point is the mean coordinate of its
  observations (`climatology.csv` lists them). Daily `temperature_2m_mean`, `temperature_2m_min`,
  `precipitation_sum` (time zone America/New_York) for Sep 1 - Nov 30 of every year 2018-2025 at every
  occupied cell come from the Open-Meteo archive API (ERA5 / ERA5-Land). Weather data by
  [Open-Meteo.com](https://open-meteo.com/), CC BY 4.0.
- Open-Meteo usage for this build: 824 archive requests x 6.5 = 5,356 weighted calls (91 days x 3 variables
  per cell-year), plus 1,200 weighted calls spent by an aborted first attempt to get point elevations from
  Open-Meteo (its elevation endpoint counts every coordinate as one call). A 0.5 degree grid would have needed
  288 cells x 8 years x 6.5 = 14,976 weighted calls, over the free daily limit.
  Whole project through 2026-10-06, including the app-side fetches for the examples, the live check and the
  weather comparison: 929 requests, 7,007.3 weighted calls (archive 5,511.2, elevation 1,203, forecast 293.1),
  all from one machine (log: `.cache/openmeteo/usage.jsonl`; the TabPFN box runs with `PEAKWEEK_OFFLINE=1`).
- Elevation of each observation point: SRTM 90 m via the OpenTopoData public API (public domain data).
  Forecast locations use the Open-Meteo elevation API (Copernicus GLO-90); on 1,200 shared points the two
  DEMs differ by 6.5 m on average (median 4 m, 95th percentile 22 m).

## Features (`peakweek/features.py`, identical code for training rows and app rows)

All seasonal windows start on Sep 1 and end the day before the observation.

| column | meaning |
|---|---|
| species_code | species (categorical, 0-7 in `peakweek/species.py` order) |
| lat, lon, elevation_m | observation point |
| doy, daylength_h | day of year; astronomical day length (CBM model) |
| cdd20 | chilling degree days since Sep 1, sum of max(0, 20 C - Tmean) |
| frost0, frost5 | nights since Sep 1 with Tmin <= 0 C / <= 5 C |
| tmin7 | mean Tmin over the last 7 days |
| tmean14 | mean Tmean over the last 14 days |
| prcp30 | mean daily precipitation (mm/day) over the last 30 days (window clipped at Sep 1) |
| tanom | season-to-date mean Tmean minus the same cell's mean over the other years 2018-2025 (leave-one-year-out; the app uses all 8 years) |
| tmean_season | season-to-date mean Tmean (input to tanom; not a model feature) |

## Counts (features.csv)

| species | green | colored | bare | total |
|---|---:|---:|---:|---:|
| red maple | 774 | 1038 | 99 | 1911 |
| sugar maple | 374 | 501 | 68 | 943 |
| sweetgum | 91 | 88 | 1 | 180 |
| northern red oak | 194 | 298 | 40 | 532 |
| American beech | 828 | 297 | 117 | 1242 |
| black gum | 65 | 48 | 4 | 117 |
| sassafras | 201 | 164 | 12 | 377 |
| Norway maple | 930 | 412 | 6 | 1348 |
| **all** | 3457 | 2846 | 347 | 6650 |

| year | green | colored | bare | total |
|---|---:|---:|---:|---:|
| 2018 | 33 | 19 | 1 | 53 |
| 2019 | 124 | 65 | 3 | 192 |
| 2020 | 257 | 309 | 5 | 571 |
| 2021 | 262 | 127 | 4 | 393 |
| 2022 | 217 | 151 | 14 | 382 |
| 2023 | 434 | 283 | 35 | 752 |
| 2024 | 1341 | 1030 | 211 | 2582 |
| 2025 | 789 | 862 | 74 | 1725 |
| **all** | 3457 | 2846 | 347 | 6650 |

## nearby_cells.json (the daily site's local check)

For each of the 103 grid cells the daily site publishes (cells with training observations), the number of
research-grade, verifiable iNaturalist observations of each of the 8 species within 50 km of the cell's weather
point (`api.inaturalist.org/v1/observations/species_counts`, one keyless call per cell, fetched once on
2026-10-06 by `scripts/build_nearby.py`). Aggregate counts only; the daily build reads this file and never
calls iNaturalist.

## Attribution

Observations: iNaturalist contributors, each under the license in `license_code`
(see `https://www.inaturalist.org/observations/<obs_id>`). Weather: Open-Meteo.com (CC BY 4.0), ERA5 by
the Copernicus Climate Change Service. Elevation: NASA SRTM via OpenTopoData.
