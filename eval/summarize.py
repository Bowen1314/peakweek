"""Recompute metrics from saved per-row predictions (eval/preds/) and print Markdown tables.

  python eval/summarize.py --split val
  python eval/summarize.py --split test --tabpfn tabpfn --baseline hgb   # CI: tabpfn minus baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from peakweek import metrics  # noqa: E402
from peakweek.species import BY_TAXON, SPECIES  # noqa: E402

PREDS = ROOT / "eval" / "preds"
SUBSET: set | None = None  # restrict every table to these obs_ids (paired comparison on a subsample)


def load(split: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(PREDS / f"{split}_{name}.csv")
    df["observed_on"] = pd.to_datetime(df["observed_on"])
    if SUBSET is not None:
        df = df[df["obs_id"].isin(SUBSET)].sort_values("obs_id").reset_index(drop=True)
    return df


def proba(df: pd.DataFrame) -> np.ndarray:
    return df[["p_green", "p_colored", "p_bare"]].to_numpy()


def runs(split: str) -> dict:
    out = {}
    path = ROOT / "eval" / "runs.jsonl"
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["split"] == split:
            out[r["name"]] = r  # last record per name wins
    return out


def table(split: str, names: list[str] | None = None) -> str:
    rs = runs(split)
    names = names or sorted(rs, key=lambda n: rs[n]["log_loss"])
    lines = ["| model | n | log loss | Brier | accuracy | s / 100 rows | max RSS MB | host |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for n in names:
        r = rs[n]
        df = load(split, n)
        m = metrics.summary(df["label_int"].to_numpy(), proba(df))
        lines.append(f"| {n} | {m['n']} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['accuracy']:.3f} | "
                     f"{r['sec_per_100_rows']:.2f} | {r.get('maxrss_mb', '')} | {r.get('host', '')} |")
    return "\n".join(lines)


def per_species(split: str, names: list[str]) -> str:
    header = "| species | n | " + " | ".join(names) + " |"
    lines = [header, "|---|---:|" + "---:|" * len(names)]
    res = {}
    for n in names:
        df = load(split, n)
        res[n] = metrics.per_group_log_loss(df["label_int"].to_numpy(), proba(df), df["taxon_id"].to_numpy())
    for s in SPECIES:
        t = s["taxon_id"]
        if t not in res[names[0]]:
            continue
        row = f"| {s['common']} | {res[names[0]][t]['n']} | " + " | ".join(f"{res[n][t]['log_loss']:.3f}" for n in names) + " |"
        lines.append(row)
    return "\n".join(lines)


def ci(split: str, a: str, b: str, n_boot: int = 2000) -> dict:
    A, B = load(split, a), load(split, b)
    assert (A["obs_id"].to_numpy() == B["obs_id"].to_numpy()).all()
    y = A["label_int"].to_numpy()
    out = {}
    for metric, fn in (("log_loss", metrics.rowwise_log_loss), ("brier", metrics.rowwise_brier)):
        la, lb = fn(y, proba(A)), fn(y, proba(B))
        week = A["observed_on"].dt.isocalendar().week.astype(str)
        clusters = A["cell_id"].astype(str) + "_w" + week
        out[metric] = {"rows": metrics.paired_bootstrap(la, lb, n_boot=n_boot, seed=0),
                       "cell_x_week": metrics.paired_bootstrap(la, lb, n_boot=n_boot, seed=0, clusters=clusters)}
    return out


def ci_markdown(split: str, a: str, b: str) -> str:
    r = ci(split, a, b)
    lines = [f"Paired bootstrap, {a} minus {b} (negative = {a} better), 2000 resamples:", "",
             "| metric | resampling unit | units | mean diff | 95% CI | P(diff<0) |", "|---|---|---:|---:|---|---:|"]
    for metric, d in r.items():
        for unit, s in d.items():
            lines.append(f"| {metric} | {unit} | {s['n_units']} | {s['mean_diff']:+.4f} | "
                         f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] | {s['p_a_better']:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--species", nargs="*", help="models for the per-species table")
    ap.add_argument("--tabpfn")
    ap.add_argument("--baseline")
    ap.add_argument("--subset-of", help="restrict all models to the obs_ids of this run's predictions")
    args = ap.parse_args()
    global SUBSET
    if args.subset_of:
        SUBSET = set(pd.read_csv(PREDS / f"{args.split}_{args.subset_of}.csv")["obs_id"])
    print(table(args.split, args.names))
    if args.species:
        print()
        print(per_species(args.split, args.species))
    if args.tabpfn and args.baseline:
        print()
        print(ci_markdown(args.split, args.tabpfn, args.baseline))


if __name__ == "__main__":
    main()
