import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.feature_extraction import FeatureHasher
from config import config
import random


def build_features(df, target_col='tidy_cod', label_col='icd10h_category', hash_dim=1<<16):
    token_cols = [c for c in df.columns if c.startswith(f'{target_col}__')]
    tokens = df[token_cols].apply(lambda r: sum(r.tolist(), []), axis=1).tolist()
    X = FeatureHasher(n_features=hash_dim, input_type='string', alternate_sign=False).transform(tokens)
    classes = sorted(df[label_col].dropna().unique().tolist())
    class_to_id = {c:i for i,c in enumerate(classes)}
    y = df[label_col].map(class_to_id).fillna(-1).astype(int).to_numpy()
    return X, y, classes

class HashedCSRDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.tocsr()
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        x = torch.from_numpy(self.X.getrow(i).toarray().ravel()).float()
        y = torch.tensor(self.y[i], dtype=torch.long)
        return x, y

def make_loaders(X, y, data_idx, config):
    g = torch.Generator().manual_seed(int(config.get('seed', 42)))
    ds = HashedCSRDataset(X, y)

    idx_train = list(data_idx['labeled'])
    idx_test  = list(data_idx['test'])
    n = len(ds)
    if idx_train and (min(idx_train) < 0 or max(idx_train) >= n):
        raise IndexError("train indices out of range")
    if idx_test and (min(idx_test) < 0 or max(idx_test) >= n):
        raise IndexError("test indices out of range")

    train_ds = Subset(ds, idx_train)
    test_ds  = Subset(ds, idx_test)

    bs = int(config.get('batch_size', 128))
    num_workers = int(config.get('dataloader_workers', 0))
    pin_memory = bool(config.get('pin_memory', False))
    prefetch_factor = config.get('prefetch_factor', 2)
    persistent_workers = bool(config.get('persistent_workers', num_workers > 0))
    test_bs = int(config.get('test_batch_size', max(bs, 2 * bs)))

    def _seed_worker(worker_id):
        s = int(config.get('seed', 42)) + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)

    dl_common = dict(num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        dl_common.update(
            worker_init_fn=_seed_worker,
            persistent_workers=persistent_workers,
            prefetch_factor=int(prefetch_factor) if prefetch_factor is not None else 2,
        )

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, generator=g, **dl_common
    )
    test_loader = DataLoader(
        test_ds, batch_size=test_bs, shuffle=False, **dl_common
    )
    return train_loader, test_loader, ds
