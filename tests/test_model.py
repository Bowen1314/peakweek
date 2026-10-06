import os
import unittest

import numpy as np
import pandas as pd

from peakweek import model
from tests.helpers import FakeClassifier, synthetic_pool


class TestAllocation(unittest.TestCase):
    def test_alloc_sums_and_floor(self):
        sizes = np.array([1000, 500, 3, 2, 300])
        take = model._alloc(sizes, 200, floor=8)
        self.assertEqual(int(take.sum()), 200)
        self.assertEqual(take[2], 3)  # rare strata fully kept
        self.assertEqual(take[3], 2)
        self.assertTrue((take <= sizes).all())

    def test_alloc_small_pool(self):
        sizes = np.array([5, 6])
        self.assertEqual(model._alloc(sizes, 100, 8).tolist(), [5, 6])


class TestContextSelection(unittest.TestCase):
    def setUp(self):
        self.pool = synthetic_pool(3000, seed=0)

    def test_stratified_deterministic(self):
        a = model.select_stratified(self.pool, 500, seed=7)
        b = model.select_stratified(self.pool.sample(frac=1.0, random_state=3), 500, seed=7)  # order-independent
        self.assertEqual(a["obs_id"].tolist(), b["obs_id"].tolist())
        c = model.select_stratified(self.pool, 500, seed=8)
        self.assertNotEqual(a["obs_id"].tolist(), c["obs_id"].tolist())
        self.assertEqual(len(a), 500)
        self.assertEqual(a["obs_id"].nunique(), 500)

    def test_stratified_keeps_rare_strata(self):
        sel = model.select_stratified(self.pool, 300, seed=0, floor=8)
        strata_pool = self.pool.groupby(["taxon_id", "label_int"]).size()
        strata_sel = sel.groupby(["taxon_id", "label_int"]).size()
        for key, size in strata_pool.items():
            self.assertGreaterEqual(strata_sel.get(key, 0), min(size, 8))

    def test_local_deterministic_and_balanced(self):
        a = model.select_local(self.pool, 400, 40.5, -74.4, 280)
        b = model.select_local(self.pool.iloc[::-1], 400, 40.5, -74.4, 280)
        self.assertEqual(a["obs_id"].tolist(), b["obs_id"].tolist())
        self.assertEqual(a.groupby("taxon_id").size().max(), 50)

    def test_small_pool_returned_whole(self):
        small = self.pool.head(50)
        self.assertEqual(len(model.select_stratified(small, 1000, seed=0)), 50)


class TestPeakweekModel(unittest.TestCase):
    def setUp(self):
        self.pool = synthetic_pool(2000, seed=1)
        self.query = synthetic_pool(120, seed=2)

    def _run(self, **kw):
        fake = FakeClassifier()
        m = model.PeakweekModel(clf_factory=lambda: fake, n_context=300, chunk=50, **kw).fit(self.pool)
        return m, fake, m.predict_proba(self.query)

    def test_shapes_and_normalization(self):
        for strategy in ("stratified", "per_species", "local"):
            with self.subTest(strategy=strategy):
                m, fake, P = self._run(strategy=strategy)
                self.assertEqual(P.shape, (120, 3))
                np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)
                self.assertTrue((P >= 0).all())
                self.assertTrue(all(s <= 300 for s in fake.fit_sizes))

    def test_per_species_drops_species_column(self):
        m, _, _ = self._run(strategy="per_species")
        self.assertNotIn("species_code", m.columns())
        self.assertEqual(m.stats["fits"], self.query["taxon_id"].nunique())

    def test_missing_class_in_context(self):
        pool = self.pool[self.pool["label_int"] != 2]
        fake = FakeClassifier()
        m = model.PeakweekModel(clf_factory=lambda: fake, n_context=300).fit(pool)
        P = m.predict_proba(self.query)
        np.testing.assert_allclose(P[:, 2], 1.0 / (300 + 3))  # add-one mass for the absent class
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)

    def test_subsample_averaging_is_deterministic(self):
        _, _, P1 = self._run(strategy="stratified", n_subsamples=3)
        _, _, P2 = self._run(strategy="stratified", n_subsamples=3)
        np.testing.assert_array_equal(P1, P2)


@unittest.skipUnless(os.environ.get("PEAKWEEK_RUN_TABPFN") == "1", "set PEAKWEEK_RUN_TABPFN=1 to run TabPFN")
class TestRealTabPFN(unittest.TestCase):
    def test_tabpfn_v2_small(self):
        pool = synthetic_pool(300, seed=3)
        query = synthetic_pool(20, seed=4)
        m = model.PeakweekModel(n_context=200, n_estimators=1).fit(pool)
        P = m.predict_proba(query)
        self.assertEqual(P.shape, (20, 3))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
