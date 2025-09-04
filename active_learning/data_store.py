from functools import lru_cache
from config import config
from import_data import import_data
from utils import create_tokenized_df, labeled_unlabeled_test_split
from data_loader import build_features

@lru_cache(maxsize=1)
def get_df():
    df,_ = import_data()
    return create_tokenized_df(df).reset_index(drop=True)

@lru_cache(maxsize=1)
def get_features():
    df = get_df()
    X, y, classes = build_features(df,
        target_col=config.get('target_col'),
        label_col=config.get('label_col'),
        hash_dim=config.get('hash_dim', 1<<16))
    return X, y, classes

def make_initial_data_idx(ordered, cfg=config, stratify=False):
    df = get_df()
    y = df[cfg.get('label_col')] if stratify else None
    labeled_idx, unlabeled_idx, test_idx = labeled_unlabeled_test_split(
        df,
        y=y,
        test_size=cfg['test_size'],
        labeled_size=cfg['labeled_size'],
        random_state=cfg['seed'],
        ordered=ordered,
    )
    return dict(
        labeled=list(labeled_idx),
        unlabeled=list(unlabeled_idx),
        test=list(test_idx),
    )