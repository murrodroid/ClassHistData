import torch
from typing import List, Dict, Tuple

def return_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device

def encode_labels(raw_labels: List[str]) -> Tuple[List[int], Dict[str, int], Dict[int, str]]:
    """
    raw_labels : e.g. ["I57", "K40", "I57", ...]
    Returns
    -------
    encoded            : List[int] mapped to 0..N-1
    label2id / id2label: both dicts for later use
    """
    unique = sorted(set(raw_labels))              # stable order
    label2id = {lbl: idx for idx, lbl in enumerate(unique)}
    id2label = {idx: lbl for lbl, idx in label2icd.items()}
    encoded = [label2id[lbl] for lbl in raw_labels]
    return encoded, label2id, id2label