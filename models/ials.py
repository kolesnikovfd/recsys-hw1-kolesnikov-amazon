"""Weighted implicit ALS, exact alternating normal-equation updates."""
import numpy as np
from scipy.linalg import solve
from .base import Recommender, dense_product


class IALS(Recommender):
    def __init__(self, factors=32, reg=10., alpha=20., iterations=12, seed=42):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def make_targets(self, data):
        return np.ones_like(data.counts.data, dtype=np.float64)

    def fit(self, data):
        counts = data.counts.astype(np.float64)
        targets = counts.copy()
        targets.data = self.make_targets(data)
        confidence = counts.copy()
        confidence.data = 1. + self.alpha * np.log1p(counts.data)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, .05, (counts.shape[0], self.factors))
        self.item_factors = rng.normal(0, .05, (counts.shape[1], self.factors))
        ct, pt = confidence.T.tocsr(), targets.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._update(confidence, targets, self.item_factors)
            self.item_factors = self._update(ct, pt, self.user_factors)
        return self

    def _update(self, confidence, targets, fixed):
        baseline = dense_product(fixed.T, fixed) + self.reg * np.eye(self.factors)
        out = np.zeros((confidence.shape[0], self.factors))
        for row in range(confidence.shape[0]):
            start, end = confidence.indptr[row:row+2]
            idx = confidence.indices[start:end]
            c, p = confidence.data[start:end], targets.data[start:end]
            vectors = fixed[idx]
            a = baseline + dense_product(vectors.T, (c - 1.)[:, None] * vectors)
            b = dense_product(vectors.T, c * p)
            out[row] = solve(a, b, assume_a='pos', check_finite=False)
        return out

    def score(self, users):
        return dense_product(self.user_factors[users], self.item_factors.T)
