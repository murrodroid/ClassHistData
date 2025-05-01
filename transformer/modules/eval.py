from __future__ import annotations
from pathlib import Path
from typing  import Optional, Tuple, List

import torch, numpy as np, pandas as pd
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
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
    logger: Optional[object] = None,     # e.g. WandBLogger
    top_k: int = 3,
) -> Tuple[str, np.ndarray, float, float, float,
           List[str], List[List[str]]]:
    """
    Returns
    -------
    report   : classification report (str)
    cm       : confusion matrix (np.ndarray)
    acc1     : top-1 accuracy
    acck     : top-k accuracy
    f1       : weighted F1
    top1_pred: list[str]                (len = N)
    topk_pred: list[list[str]]          (len = N, inner len = k)
    """
    model.eval()
    logits_list, labels_list = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits_list.append(model(**batch).logits.detach().cpu())
        labels_list.extend(batch["labels"].cpu())

    logits = torch.cat(logits_list)          # [N, C]
    labels = torch.tensor(labels_list)

    # ── metrics ─────────────────────────────────────────────
    acc1 = accuracy_score(labels, logits.argmax(1))
    acck = (
        top_k_accuracy_score(labels, logits, k=top_k)
        if top_k > 1 else acc1
    )
    f1   = f1_score(labels, logits.argmax(1), average="weighted")
    cm   = confusion_matrix(labels, logits.argmax(1))
    report = classification_report(
        labels,
        logits.argmax(1),
        target_names=[id2label[i] for i in sorted(id2label)],
        digits=3,
        zero_division=0,
    )

    # ── predictions (strings) ───────────────────────────────
    top1_ids  = logits.argmax(1)                             # [N]
    topk_ids  = logits.topk(top_k, dim=1).indices            # [N, k]

    top1_pred = [id2label[i.item()] for i in top1_ids]
    topk_pred = [
        [id2label[j.item()] for j in row] for row in topk_ids
    ]
    true_lbls = [id2label[i.item()] for i in labels]

    # ── optional WandB log ─────────────────────────────────
    if logger:
        logger.log({
            "test/acc_top1": acc1,
            f"test/acc_top{top_k}": acck,
            "test/f1": f1,
        })

    # ── optional disk outputs ──────────────────────────────
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "classification_report.txt").write_text(report)
        np.save(reports_dir / "confusion_matrix.npy", cm)

        # CSV with predictions
        df = pd.DataFrame(
            {"true": true_lbls,
             "top1": top1_pred,
             f"top{top_k}": [";".join(row) for row in topk_pred]}
        )
        df.to_csv(reports_dir / "predictions.csv", index=False)

    return report, cm, acc1, acck, f1, top1_pred, topk_pred
