"""Independent checks of objectives, scores, ranking and train-only metadata."""
import unittest
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from data_pipeline import build_training, evaluation_cohort
from evaluation import user_metrics
from models import ItemKNN, IALS, EASE, FM, CustomIALS
from models.base import dense_product


def tiny_data():
    events = pd.DataFrame([
        (f'u{u}', f'i{i}', 4.+(i%2), pd.Timestamp('2012-01-01')+pd.Timedelta(days=i),
         'dark chocolate cookie' if i%2 else 'organic green tea')
        for u, items in enumerate(([0,1,2], [1,2,3], [0,3,4], [2,4,5])) for i in items
    ], columns=['reviewerID','asin','overall','time','summary'])
    events.loc[len(events)] = ['u0','i5',1.,pd.Timestamp('2012-01-10'),'disappointing']
    return build_training(events, pd.Timestamp('2012-02-01'))


class Correctness(unittest.TestCase):
    def test_dense_product_against_non_blas_reference(self):
        rng = np.random.default_rng(9)
        for dtype in (np.float32, np.float64):
            a = rng.normal(size=(13, 7)).astype(dtype)
            b = rng.normal(size=(7, 11)).astype(dtype)
            np.testing.assert_allclose(dense_product(a, b),
                np.einsum('ik,kj->ij', a, b, optimize=False), rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(dense_product(a, b[:,0]),
                np.einsum('ik,k->i', a, b[:,0], optimize=False), rtol=1e-5, atol=1e-6)
        with self.assertRaises(FloatingPointError):
            dense_product(np.array([[np.nan]]), np.ones((1,1)))

    def test_metrics_known_answer(self):
        rec = np.arange(10)
        ndcg, recall = user_metrics(rec, [0, 2, 20])
        self.assertAlmostEqual(recall, 2/3)
        self.assertAlmostEqual(ndcg, (1 + .5)/(1+1/np.log2(3)+.5))
        self.assertEqual(user_metrics(rec, [30]), (0., 0.))

    def test_ease_matches_separate_constrained_ridge(self):
        data = tiny_data()
        model = EASE(reg=3.).fit(data)
        x = data.binary.toarray()
        expected = np.zeros((x.shape[1], x.shape[1]))
        for target in range(x.shape[1]):
            other = np.arange(x.shape[1]) != target
            z = x[:, other]
            expected[other, target] = np.linalg.solve(z.T@z+3*np.eye(z.shape[1]), z.T@x[:, target])
        np.testing.assert_allclose(model.weights, expected, atol=1e-12)

    def test_ials_update_matches_dense_weighted_least_squares(self):
        data = tiny_data()
        model = IALS(factors=3, reg=2., alpha=5.)
        fixed = np.random.default_rng(4).normal(size=(len(data.items), 3))
        c = data.counts.copy()
        c.data = 1+5*np.log1p(c.data)
        actual = model._update(c, data.binary, fixed)
        for u in range(len(data.users)):
            dense_c = np.ones(len(data.items))
            dense_c[c[u].indices] = c[u].data
            p = data.binary[u].toarray().ravel()
            expected = np.linalg.solve(fixed.T@(dense_c[:, None]*fixed)+2*np.eye(3), fixed.T@(dense_c*p))
            np.testing.assert_allclose(actual[u], expected, atol=1e-12)

    def test_custom_changes_target_and_uses_past_ages(self):
        data = tiny_data()
        p = CustomIALS().make_targets(data)
        self.assertTrue(np.all((p>0)&(p<1)))
        self.assertFalse(np.allclose(p, IALS().make_targets(data)))
        self.assertTrue(np.all(data.age_days.data > 0))

    def test_fm_equals_explicit_pairwise_formula_and_fast_scores(self):
        data = tiny_data()
        fm = FM(epochs=1, factors=3, batch_size=32).fit(data)
        u, i = 1, 2
        features = data.features[i]
        features = features[features != 0]
        vectors = [fm.user_embedding.weight[u]] + [fm.embedding.weight[j] for j in features]
        expected = fm.bias + fm.user_linear.weight[u,0] + sum(fm.linear.weight[j,0] for j in features)
        for a in range(len(vectors)):
            for b in range(a+1, len(vectors)):
                expected = expected + vectors[a]@vectors[b]
        actual = fm.logits(torch.tensor([u]), torch.tensor([i]))[0].item()
        self.assertAlmostEqual(actual, expected.item(), places=6)
        self.assertAlmostEqual(actual, float(fm.score([u])[0,i]), places=6)

    def test_all_models_mask_seen_and_are_deterministic(self):
        data = tiny_data()
        for cls, args in ((ItemKNN, {}), (EASE, {}), (IALS, {'iterations':2, 'factors':3}),
                          (CustomIALS, {'iterations':2, 'factors':3}), (FM, {'epochs':1, 'factors':3})):
            a, b = cls(**args).fit(data), cls(**args).fit(data)
            np.testing.assert_allclose(a.score([0,1]), b.score([0,1]))
            rec = a.recommend([0], data.seen, k=2)[0]
            self.assertEqual(len(set(rec)), 2)
            self.assertFalse(set(rec) & set(data.seen[0].indices))

    def test_jaccard_matches_sets(self):
        data = tiny_data()
        model = ItemKNN(k=5, shrink=2).fit(data)
        binary = data.binary.toarray()
        for i in range(binary.shape[1]):
            for j in range(binary.shape[1]):
                a, b = set(np.flatnonzero(binary[:,i])), set(np.flatnonzero(binary[:,j]))
                common = len(a & b)
                expected = 0. if i == j else common / len(a | b) * common / (common + 2)
                self.assertAlmostEqual(model.weights[i,j], expected)

    def test_custom_formula_rating_and_age(self):
        data = tiny_data()
        model = CustomIALS(tau_days=365.)
        target = model.make_targets(data)
        expected = (data.ratings.data-3)/2*(.5+.5*np.exp(-data.age_days.data/365))
        np.testing.assert_allclose(target, expected)
        self.assertEqual(data.binary[0,5], 0)
        self.assertEqual(data.seen[0,5], 1)

    def test_cohort_relevance_cold_start_and_seen(self):
        rows = [('u0',f'i{i:02}',5.,pd.Timestamp('2012-01-01'),'tea') for i in range(5)]
        rows += [('u1',f'i{i:02}',5.,pd.Timestamp('2012-01-01'),'tea') for i in range(5,20)]
        rows += [('u0','i06',1.,pd.Timestamp('2012-01-01'),'poor')]
        cols = ['reviewerID','asin','overall','time','summary']
        data = build_training(pd.DataFrame(rows,columns=cols),pd.Timestamp('2012-02-01'))
        future = pd.DataFrame([('u0','i10',5),('u0','i11',3),('u0','i06',5),
                               ('u0','new',5),('new','i12',5)],columns=cols[:3])
        truth, audit = evaluation_cohort(future,data)
        np.testing.assert_array_equal(truth[0], [10])
        self.assertEqual(audit['already_seen_pairs'], 1)
        self.assertEqual(audit['cold_user_pairs'], 1)
        self.assertEqual(audit['cold_item_pairs'], 1)

    def test_snapshot_rejects_future_data(self):
        events = pd.DataFrame({'time': [pd.Timestamp('2012-01-01')]})
        with self.assertRaises(AssertionError):
            build_training(events, pd.Timestamp('2011-01-01'))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
