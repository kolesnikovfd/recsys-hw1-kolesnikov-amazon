"""Rating strength and recency change the ALS preference target itself."""
import numpy as np
from .ials import IALS


class CustomIALS(IALS):
    def __init__(self, tau_days=730., **kwargs):
        super().__init__(**kwargs)
        self.tau_days = tau_days

    def make_targets(self, data):
        assert np.array_equal(data.counts.indices, data.ratings.indices)
        assert np.array_equal(data.counts.indptr, data.ratings.indptr)
        return ((data.ratings.data - 3.) / 2.) * (
            .5 + .5 * np.exp(-data.age_days.data / self.tau_days))
