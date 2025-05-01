from __future__ import annotations
import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, dataloader, device, id2label):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu())
            labels.extend(batch["labels"].cpu())

    report = classification_report(
        labels, preds,
        target_names=[id2label[i] for i in sorted(id2label)],
        digits=3
    )
    cm = confusion_matrix(labels, preds)
    return report, cm
