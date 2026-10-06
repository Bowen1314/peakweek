"""Timing / memory probe for TabPFN v2 on the target CPU box with peakweek-shaped data
(13 features incl. 1 categorical, 3 classes). Synthetic data; one configuration per process so that
max RSS is attributable. Appends a JSON line to eval/bench.jsonl.

  python eval/bench_tabpfn.py --n-ctx 1000 --n-est 4 --n-test 1 15 200
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


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if platform.system() == "Darwin" else r / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=1000)
    ap.add_argument("--n-est", type=int, default=4)
    ap.add_argument("--n-feat", type=int, default=13)
    ap.add_argument("--n-test", type=int, nargs="+", default=[1, 15, 200])
    args = ap.parse_args()
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    t_import = time.time() - t0
    rng = np.random.default_rng(0)
    d = args.n_feat
    n = args.n_ctx + max(args.n_test)
    X = rng.normal(size=(n, d))
    X[:, 0] = rng.integers(0, 8, size=n)
    y = np.argmax(X[:, 1:4] + rng.gumbel(size=(n, 3)), axis=1)
    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=args.n_est,
                                                      categorical_features_indices=[0])
    t = time.time()
    clf.fit(X[:args.n_ctx], y[:args.n_ctx])
    t_fit = time.time() - t
    res = {"n_ctx": args.n_ctx, "n_est": args.n_est, "n_feat": d, "import_s": round(t_import, 2),
           "fit_s": round(t_fit, 2), "predict_s": {}}
    for m in args.n_test:
        t = time.time()
        clf.predict_proba(X[args.n_ctx:args.n_ctx + m])
        res["predict_s"][m] = round(time.time() - t, 2)
    res["maxrss_mb"] = round(rss_mb(), 1)
    res["host"] = platform.node()
    print(json.dumps(res), flush=True)
    with open(ROOT / "eval" / "bench.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
