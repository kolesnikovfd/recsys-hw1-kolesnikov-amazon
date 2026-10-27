"""EASE: закрытое решение квадратичной задачи с нулевой диагональю."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender


class EASE(Recommender):
    def __init__(self, reg=100.):
        self.reg = reg

    def fit(self, data):
        self.x = data.binary
        gram = (self.x.T @ self.x).toarray().astype(np.float64)
        gram.flat[::gram.shape[0] + 1] += self.reg
        inverse = cho_solve(cho_factor(gram), np.eye(gram.shape[0]), check_finite=False)
        self.weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.weights, 0.)
        return self

    def score(self, users):
        return self.x[users] @ self.weights
