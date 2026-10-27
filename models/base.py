"""Общий детерминированный top-K для всех моделей."""
import numpy as np
from scipy.linalg.blas import get_blas_funcs


def dense_product(a, b):
    """BLAS product with explicit finite checks (avoids NumPy/Accelerate flag noise)."""
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise FloatingPointError('Non-finite matrix operand')
    if b.ndim == 1:
        result = get_blas_funcs('gemv', (a, b))(1., a, b)
    else:
        result = get_blas_funcs('gemm', (a, b))(1., a, b)
    if not np.isfinite(result).all():
        raise FloatingPointError('Non-finite matrix product')
    return result


class Recommender:
    def recommend(self, users, seen, k=10):
        scores = np.asarray(self.score(users), dtype=np.float64).copy()
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite model scores')
        for row, user in enumerate(users):
            known = seen.indices[seen.indptr[user]:seen.indptr[user + 1]]
            scores[row, known] = -np.inf
        if np.any(np.isfinite(scores).sum(axis=1) < k):
            raise ValueError('Fewer than K unseen candidates')
        # Stable sort: equal scores are ordered by ascending internal item ID.
        return np.argsort(-scores, axis=1, kind='stable')[:, :k]
