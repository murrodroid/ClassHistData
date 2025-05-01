# ---------------------------------------------------------------------
# Everything concerned with turning raw text + labels -----------------
# into PyTorch DataLoader objects lives here. -------------------------
# ---------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import numpy as np

# ---------------------------------------------------------------------
# 1.  Tiny plain-old-data container -----------------------------------
# ---------------------------------------------------------------------
class TextExample(Dataset):
    """
    Minimal Dataset: just indexes into two python lists.
    The heavy lifting (tokenisation, padding) is done later
    in `collate_batch`, not in __getitem__.
    """
    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels), "Texts and labels length mismatch"
        self.texts  = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, str | int]:
        return {
            "text":  self.texts[idx],
            "label": self.labels[idx],
        }

# ---------------------------------------------------------------------
# 2.  Collator --------------------------------------------------------
# ---------------------------------------------------------------------
def make_collate_fn(
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
) -> callable:
    """
    Factory that returns a function the DataLoader will call
    on a *batch* (list of dicts).  It is where we:
      • tokenize     (batched -> far faster than inside Dataset)
      • pad/truncate (max_length keeps Llama’s context under control)
      • move labels  to a tensor named `labels` because 🤗 expects that
    """
    def collate_batch(batch: List[Dict[str, str | int]]) -> Dict[str, torch.Tensor]:
        texts  = [item["text"]  for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

        encoded = tokenizer(
            texts,
            padding=True,          # dynamic padding to longest in batch
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encoded["labels"] = labels
        return encoded

    return collate_batch

# ---------------------------------------------------------------------
# 3.  Public helper ---------------------------------------------------
# ---------------------------------------------------------------------
def get_dataloaders(
        texts: List[str],
        labels: List[int],
        tokenizer_name: str,
        batch_size: int = 32,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        max_length: int = 512,
        num_workers: int = 0,
        pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Main entry point used by main.py

    Returns
    -------
    train_dl, val_dl, test_dl  : torch.utils.data.DataLoader
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # 3a.  Build one dataset object -----------------------------------
    full_ds = TextExample(texts, labels)

    # 3b.  Reproducible split -----------------------------------------
    g = torch.Generator().manual_seed(seed)
    total = len(full_ds)
    val_size   = int(total * val_split)
    test_size  = int(total * test_split)
    train_size = total - val_size - test_size
    train_ds, val_ds, test_ds = random_split(full_ds, [train_size, val_size, test_size], generator=g)

    # 3c.  Shared collate_fn ------------------------------------------
    collate_fn = make_collate_fn(tokenizer, max_length=max_length)

    # 3d.  Individual DataLoaders -------------------------------------
    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=g,                  
            worker_init_fn=lambda w_id: np.random.seed(seed + w_id),
        )

    train_dl = make_loader(train_ds, shuffle=True)
    val_dl   = make_loader(val_ds,   shuffle=False)
    test_dl  = make_loader(test_ds,  shuffle=False)
    return train_dl, val_dl, test_dl