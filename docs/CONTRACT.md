# Internal contract between the core (forecast) and the app (verdict/card/server)

`peakweek.forecast.forecast(lat: float, lon: float, place_name: str | None = None, today: datetime.date | None = None) -> dict`
(owned by the core). It returns probabilities only:

```json
{
  "place": {"name": "New Brunswick, NJ", "lat": 40.49, "lon": -74.45, "elevation_m": 20.0},
  "in_region": true,
  "region_note": "Trained on the US Northeast and Mid-Atlantic (lat 38.5-47.5, lon -80.5 to -66.9).",
  "generated_at": "2026-10-06T14:00:00Z",
  "today": "2026-10-06",
  "days": ["2026-10-06", "...", "2026-10-20"],
  "model": {"name": "TabPFN v2 classifier (open weights, CPU)", "context_rows": 1000, "seconds": 21.3,
            "training_data": "iNaturalist Leaves annotations, 2018-2025"},
  "weather": {"source": "Open-Meteo forecast API", "past_days": 92, "forecast_days": 16},
  "species": [
    {"taxon_id": 48098, "common": "red maple", "scientific": "Acer rubrum",
     "daily": [{"date": "2026-10-06", "green": 0.42, "colored": 0.53, "bare": 0.05}]}
  ]
}
```
- Probabilities per day sum to 1 (+/- 1e-6). `days` has 15 entries (today .. today+14); every species has the same 15 dates in order.
- `peakweek.verdict.annotate(result: dict) -> dict` (owned by the app) adds per species: `weekend` {green, colored, bare}
  (mean over the next Sat+Sun within `days`), `best_day`, `best_colored`, `verdict` in {"go","starting","wait","late"},
  `verdict_text`; and top-level `weekend_dates` and `headline`. It must not change the probabilities.
- The server never imports tabpfn at module import time; `--fixture path.json` serves a saved forecast() result so the UI
  runs on a machine without TabPFN.
- A saved real result lives at `examples/forecast_new_brunswick.json` once the core produces one; until then the app uses
  `examples/forecast_fixture.json` (hand-made, clearly marked synthetic in a top-level "synthetic": true field).
