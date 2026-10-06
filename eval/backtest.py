"""Backtest on year splits. Saves per-row probabilities to eval/preds/ and appends a run record
(config, n, metrics, timings, memory) to eval/runs.jsonl.

Protocol: validation = train 2018-2023, evaluate 2024 (choose config); test = train 2018-2024,
evaluate 2025 once with the locked config (eval/locked.json).

Examples:
  python eval/backtest.py --split val --model baselines
  python eval/backtest.py --split val --model tabpfn --strategy per_species --n-ctx 1000 --n-est 4
  python eval/backtest.py --split test --locked
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from peakweek import baselines, dataset, features, metrics  # noqa: E402
from peakweek.model import FEATURE_SETS, PeakweekModel  # noqa: E402

EVAL = ROOT / "eval"
PREDS = EVAL / "preds"
LOCKED = EVAL / "locked.json"

HGB_GRID = [
    {"learning_rate": lr, "max_leaf_nodes": leaves, "max_iter": it, "min_samples_leaf": 40, "l2_regularization": 1.0}
    for lr in (0.03, 0.1) for leaves in (7, 15) for it in (50, 150)
] + [{"early_stopping": True, "validation_fraction": 0.15, "n_iter_no_change": 10, "random_state": 0}]
C_GRID = (0.1, 1.0, 10.0)


def maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if platform.system() == "Darwin" else r / 1024


def save(split: str, name: str, test: pd.DataFrame, P: np.ndarray, cfg: dict, seconds: float, extra=None):
    PREDS.mkdir(parents=True, exist_ok=True)
    out = test[["obs_id", "observed_on", "taxon_id", "cell_id", "label_int"]].copy()
    out[["p_green", "p_colored", "p_bare"]] = P
    path = PREDS / f"{split}_{name}.csv"
    out.to_csv(path, index=False, float_format="%.6f")
    y = test["label_int"].to_numpy()
    rec = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(), "split": split, "name": name,
        "config": cfg, "n_train": int(extra.get("n_train", 0)) if extra else None, **metrics.summary(y, P),
        "seconds": round(seconds, 2), "sec_per_100_rows": round(100 * seconds / max(1, len(test)), 3),
        "maxrss_mb": round(maxrss_mb(), 1), "preds": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
    }
    if extra:
        rec.update({k: v for k, v in extra.items() if k != "n_train"})
    with open(EVAL / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


def run_baselines(split: str, train: pd.DataFrame, test: pd.DataFrame, locked: dict | None,
                  feature_set: str | None = None):
    y = test["label_int"].to_numpy()
    results = {}
    if feature_set:  # validation-only variant: LR / HGB grids on another feature set, names suffixed
        fl = FEATURE_SETS[feature_set]
        for C in C_GRID:
            t = time.time()
            P = baselines.LogRegAll(feature_list=fl, C=C).fit(train).predict_proba(test)
            save(split, f"logreg_{feature_set}_C{C}", test, P, {"C": C, "features": feature_set}, time.time() - t,
                 {"n_train": len(train)})
        for i, p in enumerate(HGB_GRID):
            t = time.time()
            P = baselines.HGB(feature_list=fl, **p).fit(train).predict_proba(test)
            save(split, f"hgb_{feature_set}_g{i}", test, P, dict(p, features=feature_set), time.time() - t,
                 {"n_train": len(train)})
        return results

    def fit_eval(name, mdl, cfg):
        t = time.time()
        P = mdl.fit(train).predict_proba(test)
        results[name] = save(split, name, test, P, cfg, time.time() - t, {"n_train": len(train)})

    fit_eval("climatology", baselines.Climatology(), {"smoothing": "add-one", "fallback": "species"})
    if locked is None:  # validation: small grids, every candidate recorded
        for C in C_GRID:
            fit_eval(f"logreg_calendar_C{C}", baselines.LogRegCalendar(C=C), {"C": C})
            fit_eval(f"logreg_all_C{C}", baselines.LogRegAll(C=C), {"C": C, "features": "full"})
        for i, p in enumerate(HGB_GRID):
            fit_eval(f"hgb_g{i}", baselines.HGB(**p), dict(p, features="full"))
        fit_eval("hgb_default", baselines.HGB(), {"defaults": True, "features": "full"})
        fit_eval("hgb_calendar_default", baselines.HGB(feature_list=features.CALENDAR_FEATURES),
                 {"defaults": True, "features": "calendar"})
    else:
        b = locked["baselines"]
        fit_eval("logreg_calendar", baselines.LogRegCalendar(C=b["logreg_calendar"]["C"]), b["logreg_calendar"])
        fit_eval("logreg_all", baselines.LogRegAll(C=b["logreg_all"]["C"]), b["logreg_all"])
        hp = {k: v for k, v in b["hgb"].items() if k not in ("features", "defaults")}
        fit_eval("hgb", baselines.HGB(**hp), b["hgb"])
    _ = y
    return results


def run_tabpfn(split: str, train: pd.DataFrame, test: pd.DataFrame, cfg: dict, name: str):
    m = PeakweekModel(**cfg).fit(train)
    t = time.time()
    P = m.predict_proba(test)
    sec = time.time() - t
    return save(split, name, test, P, m.config(), sec,
                {"n_train": len(train), "fits": m.stats["fits"], "n_test": len(test)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--model", choices=["baselines", "tabpfn"], default="tabpfn")
    ap.add_argument("--locked", action="store_true", help="use eval/locked.json (required for --split test)")
    ap.add_argument("--features", default="full", choices=sorted(FEATURE_SETS))
    ap.add_argument("--strategy", default="stratified", choices=["stratified", "per_species", "local"])
    ap.add_argument("--n-ctx", type=int, default=1000)
    ap.add_argument("--n-est", type=int, default=4)
    ap.add_argument("--subsamples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (timing probes)")
    ap.add_argument("--group-sample", type=int, default=0,
                    help="evaluate only K random (cell_id, ISO week) groups (for the costly local strategy)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--features-csv", default=None, help="alternative feature table (smoke tests)")
    ap.add_argument("--baseline-features", default=None, help="validation: LR/HGB grids on this feature set")
    ap.add_argument("--out-dir", default=None, help="write preds/runs here instead of eval/ (smoke tests)")
    args = ap.parse_args()
    global EVAL, PREDS
    if args.out_dir:
        EVAL = Path(args.out_dir)
        PREDS = EVAL / "preds"

    df = dataset.load_features(args.features_csv)
    train_years, eval_year = dataset.SPLITS[args.split]
    train, test = dataset.split(df, train_years, eval_year)
    if args.limit:
        test = test.sample(n=min(args.limit, len(test)), random_state=0).reset_index(drop=True)
    if args.group_sample:
        test = test.assign(week=[pd.Timestamp(d).isocalendar().week for d in test["observed_on"]])
        keys = sorted(set(zip(test["cell_id"], test["week"])))
        rng = np.random.default_rng(0)
        pick = {keys[i] for i in rng.choice(len(keys), size=min(args.group_sample, len(keys)), replace=False)}
        test = test[[k in pick for k in zip(test["cell_id"], test["week"])]].reset_index(drop=True)
        print(f"group sample: {len(pick)} of {len(keys)} (cell, week) groups, {len(test)} rows", flush=True)
    print(f"split={args.split} train={len(train)} test={len(test)}", flush=True)

    locked = None
    if args.split == "test" or args.locked:
        if not LOCKED.exists():
            raise SystemExit("eval/locked.json missing: lock the config on validation first")
        locked = json.loads(LOCKED.read_text())

    if args.model == "baselines":
        if args.baseline_features and args.split == "test":
            raise SystemExit("--baseline-features is a validation-only option")
        run_baselines(args.split, train, test, None if args.baseline_features else locked, args.baseline_features)
        return

    if locked is not None:
        if args.split == "test":
            for name, cfg in locked["tabpfn_runs"].items():
                run_tabpfn(args.split, train, test, cfg, name)
            return
        cfg = locked["tabpfn_runs"]["tabpfn"]
    else:
        cfg = {"feature_set": args.features, "strategy": args.strategy, "n_context": args.n_ctx,
               "n_estimators": args.n_est, "n_subsamples": args.subsamples, "seed": args.seed,
               "floor": args.floor}
    name = args.name or "tabpfn_{feature_set}_{strategy}_c{n_context}_e{n_estimators}_s{n_subsamples}".format(**cfg)
    if args.limit:
        name += f"_limit{args.limit}"
    if args.group_sample:
        name += f"_groups{args.group_sample}"
    run_tabpfn(args.split, train, test, cfg, name)


if __name__ == "__main__":
    main()
