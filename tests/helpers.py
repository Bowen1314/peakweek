"""Shared fakes for tests (no network, no TabPFN)."""

import datetime as dt

import numpy as np
import pandas as pd

from peakweek import features
from peakweek.species import SPECIES


class FakeClassifier:
    """Deterministic stand-in for TabPFN with the sklearn fit / predict_proba / classes_ interface."""

    def __init__(self):
        self.fit_sizes = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(np.asarray(y, dtype=int))
        self.fit_sizes.append(len(X))
        self.n_features_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=float))
        assert X.shape[1] == self.n_features_
        k = len(self.classes_)
        z = np.column_stack([np.sin(X.sum(axis=1) * 0.01 + j) for j in range(k)])
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def synthetic_pool(n=3000, seed=0, years=range(2018, 2026)):
    rng = np.random.default_rng(seed)
    years = list(years)
    taxa = [s["taxon_id"] for s in SPECIES]
    rows = []
    for i in range(n):
        y = years[i % len(years)]
        d = dt.date(y, 9, 1) + dt.timedelta(days=int(rng.integers(0, 91)))
        tid = taxa[int(rng.integers(0, len(taxa)))]
        lab = int(rng.choice(3, p=[0.5, 0.45, 0.05]))
        rows.append({"obs_id": 1000 + i, "observed_on": d, "year": y, "taxon_id": tid, "label_int": lab,
                     "label": ["green", "colored", "bare"][lab],
                     "cell_id": "40.25_-74.25", "lat": 38.5 + rng.random() * 9, "lon": -80 + rng.random() * 13,
                     "elevation_m": rng.random() * 800})
    df = pd.DataFrame(rows)
    df["species_code"] = df["taxon_id"].map({s["taxon_id"]: i for i, s in enumerate(SPECIES)}).astype(float)
    df["doy"] = [d.timetuple().tm_yday for d in df["observed_on"]]
    df["daylength_h"] = [features.daylength_hours(a, b) for a, b in zip(df["lat"], df["doy"])]
    for c in features.WEATHER_FEATURES:
        df[c] = rng.normal(size=n)
    return df


def synthetic_forecast_weather(today: dt.date, past_days=92, forecast_days=16, seed=0):
    rng = np.random.default_rng(seed)
    start = today - dt.timedelta(days=past_days)
    n = past_days + forecast_days
    dates = [start + dt.timedelta(days=i) for i in range(n)]
    tmean = np.array([22 - 0.15 * i + rng.normal() for i in range(n)])
    df = pd.DataFrame({"date": pd.to_datetime(dates), "tmean": tmean, "tmin": tmean - 6,
                       "prcp": rng.gamma(0.5, 4.0, size=n)})
    df.loc[n - 1, ["tmean", "tmin", "prcp"]] = np.nan  # forecast's last day is often incomplete
    return df
