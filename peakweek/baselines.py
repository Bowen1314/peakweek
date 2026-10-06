"""Baselines on the same splits as TabPFN. All return an (n, 3) probability matrix in LABELS order."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from . import config, features

K = len(config.LABELS)


def _iso_week(dates) -> np.ndarray:
    return np.array([pd.Timestamp(d).isocalendar().week for d in dates])


def _full_proba(clf, X, n_classes: int = K) -> np.ndarray:
    P = np.zeros((len(X), n_classes))
    P[:, np.asarray(clf.classes_, dtype=int)] = clf.predict_proba(X)
    return P


class Climatology:
    """Species x ISO-week class frequencies with add-one smoothing; species-level fallback when the
    (species, week) cell has no training rows."""

    name = "climatology"

    def fit(self, df: pd.DataFrame):
        y = df["label_int"].to_numpy()
        wk = _iso_week(df["observed_on"])
        self.table = {}
        for (sp, w), idx in pd.DataFrame({"sp": df["taxon_id"], "w": wk}).groupby(["sp", "w"]).groups.items():
            self.table[(sp, w)] = np.bincount(y[idx], minlength=K)
        self.species = {sp: np.bincount(y[idx], minlength=K)
                        for sp, idx in pd.DataFrame({"sp": df["taxon_id"]}).groupby("sp").groups.items()}
        self.prior = np.bincount(y, minlength=K)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        wk = _iso_week(df["observed_on"])
        out = np.zeros((len(df), K))
        for i, (sp, w) in enumerate(zip(df["taxon_id"], wk)):
            c = self.table.get((sp, w))
            if c is None:
                c = self.species.get(sp, self.prior)
            c = np.asarray(c, dtype=float) + 1.0
            out[i] = c / c.sum()
        return out


class LogRegCalendar:
    """Multinomial logistic regression on species one-hot + doy + lat (numeric standardized)."""

    name = "logreg_calendar"

    def __init__(self, C: float = 1.0):
        self.C = C

    def _pipe(self):
        pre = ColumnTransformer([
            ("sp", OneHotEncoder(handle_unknown="ignore"), ["taxon_id"]),
            ("num", StandardScaler(), ["doy", "lat"]),
        ])
        return make_pipeline(pre, LogisticRegression(C=self.C, max_iter=2000))

    def fit(self, df):
        self.m = self._pipe().fit(df[["taxon_id", "doy", "lat"]], df["label_int"])
        return self

    def predict_proba(self, df):
        return _full_proba(self.m, df[["taxon_id", "doy", "lat"]])


class LogRegAll:
    """Logistic regression on species one-hot + all numeric features (median-imputed, standardized)."""

    name = "logreg_all"

    def __init__(self, feature_list=None, C: float = 1.0):
        self.features = [f for f in (feature_list or features.FEATURES_FULL) if f != "species_code"]
        self.C = C

    def fit(self, df):
        pre = ColumnTransformer([
            ("sp", OneHotEncoder(handle_unknown="ignore"), ["taxon_id"]),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), self.features),
        ])
        self.m = make_pipeline(pre, LogisticRegression(C=self.C, max_iter=5000))
        self.m.fit(df[["taxon_id"] + self.features], df["label_int"])
        return self

    def predict_proba(self, df):
        return _full_proba(self.m, df[["taxon_id"] + self.features])


class HGB:
    """sklearn HistGradientBoostingClassifier on all features (species as a native categorical)."""

    name = "hgb"

    def __init__(self, feature_list=None, **params):
        self.features = list(feature_list or features.FEATURES_FULL)
        self.params = params

    def fit(self, df):
        cat = [f in features.CATEGORICAL for f in self.features]
        params = dict({"random_state": 0}, **self.params)
        self.m = HistGradientBoostingClassifier(categorical_features=cat, **params)
        self.m.fit(df[self.features].to_numpy(dtype=float), df["label_int"].to_numpy())
        return self

    def predict_proba(self, df):
        return _full_proba(self.m, df[self.features].to_numpy(dtype=float))


def week_of(d: dt.date) -> int:
    return pd.Timestamp(d).isocalendar().week
