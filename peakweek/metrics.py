"""Probabilistic metrics and paired bootstrap confidence intervals (numpy only)."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15


def _as_proba(P) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.clip(P, EPS, 1.0)
    return P / P.sum(axis=1, keepdims=True)


def rowwise_log_loss(y, P) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    P = _as_proba(P)
    return -np.log(P[np.arange(len(y)), y])


def rowwise_brier(y, P) -> np.ndarray:
    """Multiclass Brier per row: sum_k (p_k - 1[y=k])^2 (range 0..2, same as sklearn's multiclass)."""
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)
    onehot = np.eye(P.shape[1])[y]
    return ((P - onehot) ** 2).sum(axis=1)


def log_loss(y, P) -> float:
    return float(rowwise_log_loss(y, P).mean())


def brier(y, P) -> float:
    return float(rowwise_brier(y, P).mean())


def accuracy(y, P) -> float:
    y = np.asarray(y, dtype=int)
    return float((np.asarray(P).argmax(axis=1) == y).mean())


def summary(y, P) -> dict:
    return {"n": int(len(y)), "log_loss": log_loss(y, P), "brier": brier(y, P), "accuracy": accuracy(y, P)}


def per_group_log_loss(y, P, groups) -> dict:
    ll = rowwise_log_loss(y, P)
    s = pd.Series(ll).groupby(np.asarray(groups)).agg(["mean", "size"])
    return {k: {"log_loss": float(r["mean"]), "n": int(r["size"])} for k, r in s.iterrows()}


def paired_bootstrap(loss_a, loss_b, n_boot: int = 2000, seed: int = 0, clusters=None) -> dict:
    """CI for mean(loss_a - loss_b). Negative = model a better.

    With clusters=None, observations are resampled. Otherwise whole clusters are resampled with
    replacement and the statistic is sum(diff in drawn clusters) / count(rows in drawn clusters)."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    if clusters is None:
        idx = rng.integers(0, n, size=(n_boot, n))
        stats = d[idx].mean(axis=1)
        n_units = n
    else:
        codes, uniq = pd.factorize(pd.Series(np.asarray(clusters)).astype(str))
        k = len(uniq)
        sums = np.bincount(codes, weights=d, minlength=k)
        cnts = np.bincount(codes, minlength=k).astype(float)
        draw = rng.integers(0, k, size=(n_boot, k))
        stats = sums[draw].sum(axis=1) / cnts[draw].sum(axis=1)
        n_units = k
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_a_better": float((stats < 0).mean()), "n_units": int(n_units), "n_boot": n_boot}
