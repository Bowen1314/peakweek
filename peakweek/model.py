"""TabPFN v2 wrapper and context (in-context training set) selection.

TabPFN conditions on a small "context" of labeled rows at prediction time. Memory on the target box
(800 MB cap) limits the context to ~1000 rows, so the choice of rows matters. Strategies:

- "stratified":  one global context, stratified by species x class with a per-stratum floor so rare
                 strata (e.g. bare sweetgum) are present.
- "per_species": one context per species (stratified by class with a floor); species column dropped.
- "local":       per query group, the nearest rows per species in standardized (lat, lon, doy) space.

All selection functions are deterministic given (pool, query, strategy, seed).
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config, features

K = len(config.LABELS)

FEATURE_SETS = {
    "full": features.FEATURES_FULL,
    "calendar": features.CALENDAR_FEATURES,
    "full_no_anom": [f for f in features.FEATURES_FULL if f != "tanom"],
    "full_no_prcp": [f for f in features.FEATURES_FULL if f != "prcp30"],
    "core": features.CALENDAR_FEATURES + ["cdd20", "frost5", "tmin7", "tanom"],
}

TABPFN_NAME = "TabPFN v2 classifier (open weights, CPU)"
MIN_SPECIES_POOL = 20


# ----------------------------------------------------------------------------- context selection

def _alloc(sizes: np.ndarray, n: int, floor: int) -> np.ndarray:
    """Allocate n slots to strata: min(size, floor) each, the rest proportional to remaining size
    (largest-remainder rounding, ties broken by stratum order)."""
    sizes = np.asarray(sizes, dtype=int)
    if sizes.sum() <= n:
        return sizes.copy()
    base = np.minimum(sizes, floor)
    if base.sum() >= n:  # floors alone exceed n: shrink floors evenly
        base = np.zeros_like(sizes)
        rem = n
        while rem > 0:
            for i in np.argsort(-sizes, kind="stable"):
                if rem == 0:
                    break
                if base[i] < sizes[i]:
                    base[i] += 1
                    rem -= 1
        return base
    rest = sizes - base
    left = n - base.sum()
    quota = rest / rest.sum() * left
    extra = np.floor(quota).astype(int)
    short = left - extra.sum()
    order = np.argsort(-(quota - extra), kind="stable")
    extra[order[:short]] += 1
    return base + np.minimum(extra, rest)


def select_stratified(pool: pd.DataFrame, n: int, seed: int = 0, floor: int = 8,
                      strata: Sequence[str] = ("taxon_id", "label_int")) -> pd.DataFrame:
    """Deterministic stratified sample of up to n rows (floor per stratum)."""
    pool = pool.sort_values("obs_id").reset_index(drop=True)
    if len(pool) <= n:
        return pool
    rng = np.random.default_rng(seed)
    groups = pool.groupby(list(strata), sort=True).indices
    keys = sorted(groups)
    sizes = np.array([len(groups[k]) for k in keys])
    take = _alloc(sizes, n, floor)
    idx = []
    for k, t in zip(keys, take):
        members = np.asarray(groups[k])
        idx.extend(members[rng.permutation(len(members))[:t]])
    return pool.iloc[np.sort(idx)].reset_index(drop=True)


def select_local(pool: pd.DataFrame, n: int, query_lat: float, query_lon: float, query_doy: float,
                 scales=(1.0, 1.0, 7.0)) -> pd.DataFrame:
    """Nearest rows per species in (lat deg, lon deg * cos(lat), doy/7) space, n // n_species each.
    Deterministic: ties broken by obs_id."""
    pool = pool.sort_values("obs_id").reset_index(drop=True)
    species = sorted(pool["taxon_id"].unique())
    per = max(1, n // max(1, len(species)))
    coslat = np.cos(np.radians(query_lat))
    d = np.sqrt(((pool["lat"] - query_lat) / scales[0]) ** 2
                + ((pool["lon"] - query_lon) * coslat / scales[1]) ** 2
                + ((pool["doy"] - query_doy) / scales[2]) ** 2).to_numpy()
    idx = []
    for sp in species:
        m = np.where(pool["taxon_id"].to_numpy() == sp)[0]
        order = m[np.lexsort((pool["obs_id"].to_numpy()[m], d[m]))]
        idx.extend(order[:per])
    return pool.iloc[np.sort(idx)].reset_index(drop=True)


# ----------------------------------------------------------------------------- classifier

def tabpfn_factory(n_estimators: int = 4, seed: int = 0, categorical: Sequence[int] = ()) -> Callable:
    """Factory for the TabPFN v2 classifier (imported lazily; open v2 weights only)."""
    def make():
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device="cpu", n_estimators=n_estimators, random_state=seed,
            categorical_features_indices=list(categorical) or None,
        )
    return make


class PeakweekModel:
    """Context-selection strategy + a classifier factory. fit() stores the pool; predict_proba()
    selects contexts and calls the classifier (chunks of <= chunk rows)."""

    def __init__(self, feature_set: str = "full", strategy: str = "stratified", n_context: int = 1000,
                 n_estimators: int = 4, n_subsamples: int = 1, seed: int = 0, floor: int = 8,
                 chunk: int = 200, clf_factory: Callable | None = None, local_group_cols=("cell_id", "week")):
        self.feature_set = feature_set
        self.strategy = strategy
        self.n_context = n_context
        self.n_estimators = n_estimators
        self.n_subsamples = n_subsamples
        self.seed = seed
        self.floor = floor
        self.chunk = chunk
        self._factory = clf_factory
        self.local_group_cols = local_group_cols
        self.stats = {"fits": 0, "fit_predict_seconds": 0.0, "predicted_rows": 0}

    # features actually passed to the classifier for a given strategy
    def columns(self) -> list[str]:
        cols = list(FEATURE_SETS[self.feature_set])
        if self.strategy == "per_species":
            cols = [c for c in cols if c != "species_code"]
        return cols

    def config(self) -> dict:
        return {"feature_set": self.feature_set, "features": self.columns(), "strategy": self.strategy,
                "n_context": self.n_context, "n_estimators": self.n_estimators,
                "n_subsamples": self.n_subsamples, "seed": self.seed, "floor": self.floor}

    def _make(self, seed: int):
        cols = self.columns()
        cat = [i for i, c in enumerate(cols) if c in features.CATEGORICAL]
        if self._factory is not None:
            return self._factory()
        return tabpfn_factory(self.n_estimators, seed, cat)()

    def fit(self, pool: pd.DataFrame):
        self.pool = pool.reset_index(drop=True)
        return self

    def _fit_predict(self, ctx: pd.DataFrame, query: pd.DataFrame, seed: int) -> np.ndarray:
        cols = self.columns()
        t0 = time.time()
        clf = self._make(seed)
        clf.fit(ctx[cols].to_numpy(dtype=float), ctx["label_int"].to_numpy(dtype=int))
        classes = np.asarray(clf.classes_, dtype=int)
        out = np.zeros((len(query), K))
        X = query[cols].to_numpy(dtype=float)
        for i in range(0, len(X), self.chunk):
            out[i:i + self.chunk][:, classes] = clf.predict_proba(X[i:i + self.chunk])
        missing = [k for k in range(K) if k not in set(classes.tolist())]
        if missing:  # a class absent from the context gets add-one mass instead of exactly 0
            eps = 1.0 / (len(ctx) + K)
            out[:, missing] = eps
            out[:, classes] *= (1.0 - eps * len(missing)) / out[:, classes].sum(axis=1, keepdims=True)
        self.stats["fits"] += 1
        self.stats["fit_predict_seconds"] += time.time() - t0
        self.stats["predicted_rows"] += len(query)
        return out

    def contexts(self, query: pd.DataFrame) -> list[tuple[np.ndarray, pd.DataFrame, int]]:
        """[(query row positions, context frame, seed)] covering every query row once per subsample."""
        jobs = []
        q = query.reset_index(drop=True)
        for s in range(self.n_subsamples):
            seed = self.seed + s
            if self.strategy == "stratified":
                ctx = select_stratified(self.pool, self.n_context, seed, self.floor)
                jobs.append((np.arange(len(q)), ctx, seed))
            elif self.strategy == "per_species":
                for sp in sorted(q["taxon_id"].unique()):
                    pos = np.where(q["taxon_id"].to_numpy() == sp)[0]
                    sp_pool = self.pool[self.pool["taxon_id"] == sp]
                    if len(sp_pool) < MIN_SPECIES_POOL:  # too few rows of this species: global context
                        ctx = select_stratified(self.pool, self.n_context, seed, self.floor)
                    else:
                        ctx = select_stratified(sp_pool, self.n_context, seed, self.floor, strata=("label_int",))
                    jobs.append((pos, ctx, seed))
            elif self.strategy == "local":
                keys = list(zip(*(q[c].tolist() for c in self.local_group_cols)))
                groups: dict = {}
                for i, k in enumerate(keys):
                    groups.setdefault(k, []).append(i)
                for key in sorted(groups):
                    pos = np.asarray(groups[key])
                    g = q.iloc[pos]
                    ctx = select_local(self.pool, self.n_context, float(g["lat"].mean()),
                                       float(g["lon"].mean()), float(g["doy"].mean()))
                    jobs.append((pos, ctx, seed))
            else:
                raise ValueError(f"unknown strategy {self.strategy}")
        return jobs

    def predict_proba(self, query: pd.DataFrame) -> np.ndarray:
        q = query.reset_index(drop=True)
        if self.strategy == "local" and "week" in self.local_group_cols and "week" not in q:
            q = q.assign(week=[pd.Timestamp(d).isocalendar().week for d in q["observed_on"]])
        out = np.zeros((len(q), K))
        for pos, ctx, seed in self.contexts(q):
            out[pos] += self._fit_predict(ctx, q.iloc[pos], seed)
        out /= self.n_subsamples
        return out / out.sum(axis=1, keepdims=True)
