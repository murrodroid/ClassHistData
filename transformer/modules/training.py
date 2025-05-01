# ------------------------------------------------------------------
# Handles:  optimiser, scheduler, epoch/step loops, metrics, 
#           gradient clipping, mixed precision (optional) and
#           checkpointing the best model. 
# ------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup

# ---------------------------------------------------------------
def create_optimizer_and_scheduler(
    model,
    learning_rate: float,
    num_epochs: int,
    train_dataloader,
    weight_decay: float = 0.01,
):
    """
    Classic setup: AdamW + linear warm-up/decay.
    """
    # Separate decay / no-decay parameters (bias & LayerNorm -> no decay)
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(grouped_parameters, lr=learning_rate)

    steps_per_epoch = len(train_dataloader)
    num_training_steps = steps_per_epoch * num_epochs
    num_warmup_steps  = int(0.1 * num_training_steps)     # 10 % warm-up

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


# ---------------------------------------------------------------
def run_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    grad_clip: float,
    train: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Dict[str, float]:
    """
    One full pass over `dataloader`.
    If `train=True` we run optimisation; otherwise just eval.
    Returns a dict of aggregated metrics.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss, n_samples = 0.0, 0
    preds_all, labels_all = [], []

    progress = tqdm(dataloader, desc="train" if train else "eval", leave=False)
    for batch in progress:
        # Move batch to device ---------
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(**batch)
            loss    = outputs.loss
            logits  = outputs.logits

        if train:
            scaler.scale(loss).backward() if scaler else loss.backward()
            # Gradient clipping ----------
            if grad_clip is not None:
                if scaler:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            # Optim step + scheduler ------
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Book-keeping -------------------
        total_loss += loss.item() * batch["labels"].size(0)
        n_samples  += batch["labels"].size(0)

        preds = logits.argmax(dim=-1).detach().cpu()
        labels = batch["labels"].detach().cpu()
        preds_all.extend(preds)
        labels_all.extend(labels)

    # Metrics --------------------------
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="weighted")

    return {
        "loss": total_loss / n_samples,
        "acc":  acc,
        "f1":   f1,
    }


# ---------------------------------------------------------------
def train_model(
    model,
    train_dl,
    val_dl,
    device,
    *,
    num_epochs: int,
    learning_rate: float,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    checkpoint_dir: str | Path = "checkpoints",
) -> Dict[str, List[float]]:
    """
    High-level training loop. Saves the *best* (highest val-F1) weights
    to `<checkpoint_dir>/best.pt`.
    Returns history dict with loss & metrics for later plotting.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        learning_rate,
        num_epochs,
        train_dl,
    )
    scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_state = -1.0, None

    for epoch in range(1, num_epochs + 1):
        print(f"\n— Epoch {epoch}/{num_epochs} —")
        train_stats = run_epoch(
            model, train_dl, optimizer, scheduler,
            device, grad_clip, train=True, scaler=scaler
        )
        val_stats = run_epoch(
            model, val_dl, optimizer, scheduler,
            device, grad_clip, train=False, scaler=None
        )

        # Logging ----------------------
        print(
            f"train loss {train_stats['loss']:.4f} | "
            f"val loss {val_stats['loss']:.4f} | "
            f"val F1 {val_stats['f1']:.4f}"
        )

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])
        history["val_f1"].append(val_stats["f1"])

        # Checkpoint -------------------
        if val_stats["f1"] > best_f1:
            best_f1 = val_stats["f1"]
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "f1": best_f1,
            }
            torch.save(best_state, checkpoint_dir / "best.pt")
            print("⇢ New best model saved")

    return history
