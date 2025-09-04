import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.feature_extraction import FeatureHasher

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

def make_loaders(X, y, data_idx, batch_size=128, seed=42):
    g = torch.Generator().manual_seed(seed)
    ds = HashedCSRDataset(X, y)
    train_loader = DataLoader(Subset(ds, data_idx['labeled']), batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(Subset(ds, data_idx['test']), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, ds
