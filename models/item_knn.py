"""Item neighbourhoods based on shrunken Jaccard similarity."""
import numpy as np
from scipy import sparse
from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, k=30, shrink=10):
        self.k, self.shrink = k, shrink

    def fit(self, data):
        self.x = data.binary
        intersections = (self.x.T @ self.x).toarray().astype(float)
        sizes = np.diag(intersections).copy()
        unions = sizes[:, None] + sizes[None, :] - intersections
        sim = intersections / np.maximum(unions, 1.)
        sim *= intersections / np.maximum(intersections + self.shrink, 1e-12)
        np.fill_diagonal(sim, 0.)
        neighbours = np.argsort(-sim, axis=0, kind='stable')[:min(self.k, len(sizes)-1)]
        columns = np.broadcast_to(np.arange(len(sizes)), neighbours.shape)
        self.weights = sparse.csc_matrix((sim[neighbours, columns].ravel(),
            (neighbours.ravel(), columns.ravel())), shape=sim.shape)
        return self

    def score(self, users):
        return (self.x[users] @ self.weights).toarray()
