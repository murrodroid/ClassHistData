# modules/eval.py
from __future__ import annotations
from pathlib import Path
from typing  import Optional, Tuple

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

@torch.inference_mode()
def evaluate(
    model,
    dataloader,
    device,
    id2label: dict[int, str],
    *,
    reports_dir: Path | None = None,
    logger: Optional[object] = None,   # e.g. WandBLogger
) -> Tuple[str, np.ndarray, float, float]:
    """
    Runs the model on `dataloader` and returns:

    1. classification report (string)
    2. confusion matrix  (np.ndarray)
    3. accuracy (float)
    4. weighted-F1 (float)

    If `reports_dir` is supplied the report and confusion matrix are
    saved to disk.  If `logger` is supplied metrics go to WandB.
    """
    model.eval()
    preds:  list[int] = []
    labels: list[int] = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())

    # ---------------- metrics ----------------
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    cm  = confusion_matrix(labels, preds)
    report = classification_report(
        labels,
        preds,
        target_names=[id2label[i] for i in sorted(id2label)],
        digits=3,
        zero_division=0,
    )

    # ---------------- logging ----------------
    if logger:
        logger.log({"test/acc": acc, "test/f1": f1})

    # ---------------- save to disk -----------
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "classification_report.txt").write_text(report)
        np.save(reports_dir / "confusion_matrix.npy", cm)

    return report, cm, acc, f1