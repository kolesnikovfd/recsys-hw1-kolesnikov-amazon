"""Full-catalog metrics, macro averaging and paired bootstrap."""
import numpy as np
import pandas as pd


def user_metrics(recommended, relevant, k=10):
    hit = np.isin(recommended[:k], relevant).astype(float)
    discounts = 1. / np.log2(np.arange(2, k+2))
    return float(hit @ discounts / discounts[:min(k, len(relevant))].sum()), float(hit.sum()/len(relevant))


def evaluate(model, data, truth, k=10, batch_size=128):
    users = np.array(sorted(truth), dtype=int)
    rows, all_recs = [], []
    for start in range(0, len(users), batch_size):
        batch = users[start:start+batch_size]
        recommendations = model.recommend(batch, data.seen, k)
        for user, rec in zip(batch, recommendations):
            ndcg, recall = user_metrics(rec, truth[user], k)
            rows.append((user, ndcg, recall))
        all_recs.append(recommendations)
    if not rows:
        raise ValueError('Empty evaluation cohort')
    per_user = pd.DataFrame(rows, columns=['user', 'NDCG@10', 'Recall@10']).set_index('user')
    result = per_user.mean().to_dict()
    result['Coverage@10'] = len(np.unique(np.concatenate(all_recs))) / len(data.items)
    return result, per_user


def paired_bootstrap(a, b, seed=42, repeats=2000):
    assert a.index.equals(b.index)
    delta = (a - b).to_numpy()
    rng = np.random.default_rng(seed)
    values = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(repeats)])
    return {'mean_delta': float(delta.mean()), 'ci_low': float(np.quantile(values, .025)),
            'ci_high': float(np.quantile(values, .975))}
