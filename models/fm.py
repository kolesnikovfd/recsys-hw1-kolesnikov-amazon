"""Second-order FM with item metadata; manual model, PyTorch autograd."""
import numpy as np
import torch
from torch import nn
from .base import Recommender, dense_product


class FM(Recommender):
    def __init__(self, factors=32, lr=.01, weight_decay=1e-5, epochs=8,
                 negatives=3, batch_size=4096, metadata=True, seed=42):
        self.factors, self.lr, self.weight_decay = factors, lr, weight_decay
        self.epochs, self.negatives, self.batch_size = epochs, negatives, batch_size
        self.metadata, self.seed = metadata, seed

    def _item_terms(self, items):
        features = self.features[items]
        mask = (features != 0).float()
        v = self.embedding(features) * mask[..., None]
        summed = v.sum(dim=1)
        pairwise = .5 * (summed.square() - v.square().sum(dim=1)).sum(dim=1)
        linear = (self.linear(features).squeeze(-1) * mask).sum(dim=1)
        return summed, linear + pairwise

    def logits(self, users, items):
        vectors, item_bias = self._item_terms(items)
        return self.bias + self.user_linear(users).squeeze(-1) + item_bias + (
            self.user_embedding(users) * vectors).sum(dim=1)

    def fit(self, data):
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(self.seed)
        nu, ni = data.binary.shape
        features = data.features if self.metadata else data.features[:, :1]
        self.features = torch.as_tensor(features, dtype=torch.long)
        self.embedding = nn.Embedding(int(features.max())+1, self.factors, padding_idx=0)
        self.linear = nn.Embedding(int(features.max())+1, 1, padding_idx=0)
        self.user_embedding = nn.Embedding(nu, self.factors)
        self.user_linear = nn.Embedding(nu, 1)
        for emb in (self.embedding, self.user_embedding):
            nn.init.normal_(emb.weight, std=.03)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.user_linear.weight)
        self.bias = nn.Parameter(torch.zeros(()))
        parameters = [self.bias] + [p for m in (self.embedding, self.linear,
            self.user_embedding, self.user_linear) for p in m.parameters()]
        optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        positives_u, positives_i = data.binary.nonzero()
        known = data.seen.toarray().astype(bool)
        if np.any(known.sum(axis=1) == ni):
            raise ValueError('Negative sampling requires an unseen item for each user')
        self.loss_history = []
        for _ in range(self.epochs):
            neg_u = np.repeat(positives_u, self.negatives)
            neg_i = rng.integers(ni, size=len(neg_u))
            invalid = known[neg_u, neg_i]
            while invalid.any():
                neg_i[invalid] = rng.integers(ni, size=int(invalid.sum()))
                invalid = known[neg_u, neg_i]
            users = np.concatenate([positives_u, neg_u])
            items = np.concatenate([positives_i, neg_i])
            labels = np.concatenate([np.ones(len(positives_u)), np.zeros(len(neg_u))]).astype('float32')
            order = rng.permutation(len(users))
            total = 0.
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start+self.batch_size]
                logits = self.logits(torch.as_tensor(users[batch], dtype=torch.long),
                                     torch.as_tensor(items[batch], dtype=torch.long))
                loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels[batch]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            self.loss_history.append(total / len(users))
        # Exact algebraic decomposition of the FM for efficient full-catalog scoring.
        with torch.no_grad():
            summed, biases = self._item_terms(torch.arange(ni))
            self.item_vectors = summed.numpy().copy()
            self.item_bias = biases.numpy().copy()
            self.user_vectors = self.user_embedding.weight.detach().numpy().copy()
            self.user_bias = self.user_linear.weight.detach().numpy().ravel().copy()
            self.global_bias = self.bias.item()
        return self

    def score(self, users):
        return dense_product(self.user_vectors[users], self.item_vectors.T) + self.item_bias[None, :] + (
            self.user_bias[users, None] + self.global_bias)
