import unittest

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from peakweek import metrics


class TestMetricsVsSklearn(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = np.array([0, 1, 2, 1, 0, 0, 2, 1, 1, 0])
        P = rng.random((10, 3)) + 0.05
        self.P = P / P.sum(axis=1, keepdims=True)

    def test_log_loss(self):
        self.assertAlmostEqual(metrics.log_loss(self.y, self.P), log_loss(self.y, self.P, labels=[0, 1, 2]), places=10)

    def test_brier(self):
        ref = brier_score_loss(self.y, self.P, labels=[0, 1, 2])
        self.assertAlmostEqual(metrics.brier(self.y, self.P), ref, places=10)
        # hand sum for the first row
        row0 = sum((self.P[0, k] - (1.0 if k == self.y[0] else 0.0)) ** 2 for k in range(3))
        self.assertAlmostEqual(metrics.rowwise_brier(self.y, self.P)[0], row0, places=12)

    def test_accuracy(self):
        self.assertAlmostEqual(metrics.accuracy(self.y, self.P), accuracy_score(self.y, self.P.argmax(1)))

    def test_log_loss_clips_zero(self):
        P = np.array([[1.0, 0.0, 0.0]])
        self.assertTrue(np.isfinite(metrics.log_loss(np.array([1]), P)))

    def test_per_group(self):
        g = np.array(["a"] * 5 + ["b"] * 5)
        out = metrics.per_group_log_loss(self.y, self.P, g)
        self.assertEqual(out["a"]["n"], 5)
        self.assertAlmostEqual(out["a"]["log_loss"], log_loss(self.y[:5], self.P[:5], labels=[0, 1, 2]), places=10)


class TestBootstrap(unittest.TestCase):
    def test_paired_bootstrap_deterministic_and_brackets_mean(self):
        rng = np.random.default_rng(1)
        a = rng.random(200)
        b = a + 0.1 + rng.normal(scale=0.05, size=200)
        r1 = metrics.paired_bootstrap(a, b, n_boot=500, seed=3)
        r2 = metrics.paired_bootstrap(a, b, n_boot=500, seed=3)
        self.assertEqual(r1, r2)
        self.assertLess(r1["ci_low"], r1["mean_diff"])
        self.assertGreater(r1["ci_high"], r1["mean_diff"])
        self.assertLess(r1["ci_high"], 0)
        self.assertEqual(r1["p_a_better"], 1.0)

    def test_cluster_bootstrap_wider_with_correlated_clusters(self):
        rng = np.random.default_rng(2)
        k, m = 20, 10
        cluster_effect = np.repeat(rng.normal(scale=0.5, size=k), m)
        d = cluster_effect + rng.normal(scale=0.05, size=k * m)
        clusters = np.repeat(np.arange(k), m)
        rows = metrics.paired_bootstrap(d, np.zeros_like(d), n_boot=1000, seed=0)
        cl = metrics.paired_bootstrap(d, np.zeros_like(d), n_boot=1000, seed=0, clusters=clusters)
        self.assertEqual(cl["n_units"], k)
        self.assertGreater(cl["ci_high"] - cl["ci_low"], rows["ci_high"] - rows["ci_low"])


if __name__ == "__main__":
    unittest.main()
