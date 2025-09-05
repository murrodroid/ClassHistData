from sklearn.model_selection import train_test_split
from config import config
import numpy as np
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import random


def _fixed_test_loader(X, y, idx, config):
    from data_loader import HashedCSRDataset  # local import to avoid cycles

    ds = HashedCSRDataset(X, y)
    test_idx = [int(i) for i in idx]
    n = len(ds)
    if test_idx and (min(test_idx) < 0 or max(test_idx) >= n):
        raise IndexError("fixed_test_idx out of range")

    test_ds = Subset(ds, test_idx)

    num_workers = int(config.get('dataloader_workers', 0))
    pin_memory = bool(config.get('pin_memory', False))
    prefetch_factor = config.get('prefetch_factor', 2)
    persistent_workers = bool(config.get('persistent_workers', num_workers > 0))
    bs = int(config.get('test_batch_size', max(int(config.get('batch_size', 128)), 2*int(config.get('batch_size', 128)))))

    def _seed_worker(worker_id):
        s = int(config.get('seed', 42)) + worker_id
        random.seed(s); np.random.seed(s); torch.manual_seed(s)

    dl_common = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        dl_common.update(
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            prefetch_factor=int(prefetch_factor) if prefetch_factor is not None else 2,
        )

    return DataLoader(test_ds, batch_size=bs, shuffle=False, **dl_common)
    
def labeled_unlabeled_test_split(
    df,
    y=None,
    test_size=config['test_size'],
    labeled_size=config['labeled_size'],
    random_state=config['seed'],
    ordered=False,
):
    n = len(df)
    idx = np.arange(n)
    strat = y.values if y is not None else None

    pool_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=strat
    )

    p = min(1.0, labeled_size / max(1e-12, (1 - test_size)))  # safe div

    if ordered:
        pool_sorted = np.sort(pool_idx)
        k = int(np.floor(p * len(pool_sorted)))
        k = max(0, min(k, len(pool_sorted)))
        labeled_idx = pool_sorted[:k]
        unlabeled_idx = pool_sorted[k:]
    else:
        strat_pool = strat[pool_idx] if strat is not None else None
        unlabeled_idx, labeled_idx = train_test_split(
            pool_idx,
            test_size=p,
            random_state=random_state,
            stratify=strat_pool
        )

    return list(map(int, labeled_idx)), list(map(int, unlabeled_idx)), list(map(int, test_idx))


def update_indexes(data_idx, selected):
    s = set(selected)
    data_idx['labeled'] = list(data_idx['labeled']) + selected
    data_idx['unlabeled'] = [i for i in data_idx['unlabeled'] if i not in s]
    return data_idx


def validate_indices(data_idx, n):
    for k in ('labeled','unlabeled','test'):
        xs = list(map(int, data_idx[k]))
        if xs:
            assert min(xs) >= 0 and max(xs) < n, f'{k} index out of range'
            assert len(xs) == len(set(xs)), f'{k} has duplicates'
    a, b, c = map(set, (data_idx['labeled'], data_idx['unlabeled'], data_idx['test']))
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c), 'overlapping splits'


def create_tokenized_df(df,token_types=config['token_types'],target_col=config['target_col'],config=config):
    def norm(s):
        return re.sub(r'\s+', ' ', str(s).strip().lower())
    def char_ngrams(s, n):
        s = norm(s)
        return [s[i:i+n] for i in range(len(s)-n+1)] if n > 0 else list(s)
    def word_tokens(s):
        return re.findall(r'\b\w+\b', norm(s))
    def word_ngrams(toks, n):
        if n <= 1:
            return toks
        return [' '.join(toks[i:i+n]) for i in range(len(toks)-n+1)]

    token_types = token_types or config['token_types']
    out = df.copy()
    text = out[target_col].fillna('')

    for tt in token_types:
        method = tt.get('method')
        n = int(tt.get('ngram', 0))
        if method == 'char':
            n = max(1, n)
            col = f'{target_col}__char{n}'
            out[col] = text.apply(lambda s: char_ngrams(s, n))
        elif method == 'word':
            n = max(1, n)
            base = text.apply(word_tokens)
            col = f'{target_col}__word' if n == 1 else f'{target_col}__word{n}'
            out[col] = base if n == 1 else base.apply(lambda toks: word_ngrams(toks, n))

    return out


def set_seed(s):
    import random, numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
