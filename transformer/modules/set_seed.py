# --------------------------------------------------------------------
# One true seed for the whole project.  Import and call once in main.
# --------------------------------------------------------------------
from __future__ import annotations
import os, random, numpy as np, torch

# Optional: add any third-party libs that need a seed here
# e.g. import datasets, transformers  (HF uses np / torch seeds internally)

def set_seed(seed: int | None = 42) -> int:
    """
    * Sets every RNG we care about.
    * Turns on deterministic CUDA kernels (slower but reproducible).
    * Returns the seed so you can log it.

    Call it once *at the very top of main.py* **before** you build
    DataLoaders or models.
    """
    if seed is None:
        # Don't pick the same number twice if user passes None
        seed = int.from_bytes(os.urandom(2), "big")  # 0-65535
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure every call path is deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    return seed
