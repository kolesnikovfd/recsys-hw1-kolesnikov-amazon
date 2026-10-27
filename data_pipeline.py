"""Amazon review parsing, chronological snapshots and historical item features."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import gzip
import hashlib
import json
import os
import re
import urllib.request
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon/links.html'
ORIGINAL_URL = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food_5.json.gz'
DATA_URL = 'https://raw.githubusercontent.com/ParsProgrammer/Exp-RS/4cb769058e2e06801dcb991d7212432d88c09fc2/reviews_Grocery_and_Gourmet_Food_5.json.gz'
DATA_SHA256 = '77c5b65fb395dcd24f6a81d95d2ea471e8d9a82b70431b39919c14ab52547dd1'
VALID_START = pd.Timestamp('2013-07-01')
TEST_START = pd.Timestamp('2014-01-01')
STOP_WORDS = set('the and for this that with are was have has but not you your they them from very good great best love product these those its just really like more than some too all can one would will been much'.split())


def load_raw():
    directory = Path(os.environ.get('RECSYS_DATA_DIR', 'data'))
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / 'reviews_Grocery_and_Gourmet_Food_5.json.gz'
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(DATA_URL, timeout=90) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(f'Archive checksum mismatch: {digest}')
    # Keep only fields needed for the experiment, not reviewer names or full texts.
    fields = ['reviewerID', 'asin', 'overall', 'unixReviewTime', 'summary']
    with gzip.open(archive, 'rt', encoding='utf-8') as source:
        rows = [{key: row.get(key) for key in fields} for row in map(json.loads, source)]
    return pd.DataFrame(rows)


def clean_data(raw):
    stages = [('Исходные отзывы 5-core', len(raw))]
    frame = raw.drop_duplicates().copy()
    stages.append(('Без полных дубликатов выбранных полей', len(frame)))
    frame = frame.dropna(subset=['reviewerID', 'asin', 'overall', 'unixReviewTime'])
    frame = frame[frame.overall.isin([1, 2, 3, 4, 5])].copy()
    frame['time'] = pd.to_datetime(frame.unixReviewTime, unit='s', utc=True, errors='coerce').dt.tz_localize(None)
    frame = frame.dropna(subset=['time'])
    frame['summary'] = frame.summary.fillna('').astype(str)
    stages.append(('Валидные ID, время и оценка 1–5', len(frame)))
    # If a pair was repeated, retain its FIRST review: a later edit cannot rewrite the past.
    frame = frame.sort_values(['time', 'reviewerID', 'asin'], kind='stable')
    frame = frame.drop_duplicates(['reviewerID', 'asin'], keep='first').reset_index(drop=True)
    stages.append(('Первая оценка каждой пары пользователь–товар', len(frame)))
    assert 100_000 <= len(frame) <= 2_000_000
    assert frame.reviewerID.nunique() >= 1000 and frame.asin.nunique() >= 1000
    return frame, pd.DataFrame(stages, columns=['Этап', 'Строки'])


@dataclass
class TrainingData:
    counts: sparse.csr_matrix
    binary: sparse.csr_matrix
    age_days: sparse.csr_matrix
    ratings: sparse.csr_matrix
    seen: sparse.csr_matrix
    features: np.ndarray
    users: np.ndarray
    items: np.ndarray
    metadata: pd.DataFrame
    vocabulary: dict
    cutoff: pd.Timestamp


def build_training(events, cutoff):
    assert len(events) and events.time.max() < cutoff, 'Snapshot includes future observations'
    positive = events[events.overall >= 4].copy()
    users = np.sort(positive.reviewerID.unique())
    items = np.sort(positive.asin.unique())
    user_map, item_map = {x: j for j, x in enumerate(users)}, {x: j for j, x in enumerate(items)}
    u, i = positive.reviewerID.map(user_map).to_numpy(), positive.asin.map(item_map).to_numpy()
    shape = (len(users), len(items))
    def matrix(values, rows=u, columns=i):
        result = sparse.csr_matrix((np.asarray(values, dtype=float), (rows, columns)), shape=shape)
        result.sort_indices()
        return result
    binary = matrix(np.ones(len(positive)))
    assert np.all(binary.data == 1), 'Clean duplicate pairs before constructing a snapshot'
    age_days = matrix((cutoff-positive.time).dt.total_seconds().to_numpy()/86400)
    ratings = matrix(positive.overall.to_numpy())
    known = events[events.reviewerID.isin(user_map) & events.asin.isin(item_map)]
    seen = matrix(np.ones(len(known)), known.reviewerID.map(user_map).to_numpy(), known.asin.map(item_map).to_numpy())
    seen.data[:] = 1.
    metadata = events.groupby('asin').agg(mean_rating=('overall', 'mean'),
        review_count=('overall', 'size'), first_seen=('time', 'min'),
        summaries=('summary', lambda s: ' '.join(s))).reindex(items)
    metadata['age_days'] = (cutoff-metadata.first_seen).dt.total_seconds()/86400
    # A document is the item's already published review summaries, not its future reviews.
    documents = [Counter(w for w in re.findall(r'[a-z]{3,}', text.lower()) if w not in STOP_WORDS)
                 for text in metadata.summaries]
    frequency = Counter(w for doc in documents for w in doc)
    allowed = [w for w, n in frequency.items() if 5 <= n <= .3*len(items)]
    vocabulary_words = sorted(sorted(allowed, key=lambda w: (-frequency[w], w))[:2000])
    vocabulary = {w: len(items)+1+j for j, w in enumerate(vocabulary_words)}
    # Six words ranked by TF-IDF; three fixed-band historical attributes.
    features = np.zeros((len(items), 10), dtype=np.int64)
    features[:, 0] = np.arange(1, len(items)+1)
    for row, doc in enumerate(documents):
        words = sorted((w for w in doc if w in vocabulary), key=lambda w: (
            -(1+np.log(doc[w]))*(1+np.log((1+len(items))/(1+frequency[w]))), w))[:6]
        features[row, 1:1+len(words)] = [vocabulary[w] for w in words]
    offset = len(items)+len(vocabulary)+1
    for column, values, edges in [
        (7, metadata.mean_rating, [2, 3, 3.5, 4, 4.5]),
        (8, metadata.review_count, [2, 5, 10, 20, 50, 100]),
        (9, metadata.age_days, [30, 90, 180, 365, 730, 1460])]:
        features[:, column] = offset + np.digitize(values.to_numpy(), edges)
        offset += len(edges)+1
    return TrainingData(binary.copy(), binary, age_days, ratings, seen, features, users,
                        items, metadata, vocabulary, cutoff)


def evaluation_cohort(future, data):
    user_map = {x: j for j, x in enumerate(data.users)}
    item_map = {x: j for j, x in enumerate(data.items)}
    relevant = future[future.overall >= 4][['reviewerID', 'asin']].drop_duplicates()
    ku, ki = relevant.reviewerID.isin(user_map), relevant.asin.isin(item_map)
    warm = relevant[ku & ki].copy()
    warm['u'], warm['i'] = warm.reviewerID.map(user_map), warm.asin.map(item_map)
    seen = np.asarray(data.seen[warm.u.to_numpy(), warm.i.to_numpy()]).ravel() > 0
    novel = warm[~seen]
    truth = {int(u): np.sort(group.i.to_numpy()) for u, group in novel.groupby('u')
             if len(data.items)-data.seen[int(u)].nnz >= 10}
    audit = dict(future_reviews=len(future), future_positive_pairs=len(relevant),
        cold_user_pairs=int((~ku).sum()), cold_item_pairs=int((~ki).sum()),
        warm_pairs=len(warm), already_seen_pairs=int(seen.sum()),
        novel_relevant_pairs=sum(map(len, truth.values())), evaluated_users=len(truth),
        future_users=future.reviewerID.nunique(), catalog_items=len(data.items))
    return truth, audit
