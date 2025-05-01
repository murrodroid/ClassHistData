# -------------------------------------------------------------------
# Handles: optimiser, scheduler, AMP, gradient clipping, epoch loops,
#          WandB logging (optional) and checkpointing the best model.
# -------------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing   import Dict, List, Optional

import torch
from torch import autocast
from torch.optim            import AdamW
from torch.nn.utils         import clip_grad_norm_
from tqdm.auto              import tqdm
from sklearn.metrics        import accuracy_score, f1_score, top_k_accuracy_score
from transformers           import get_linear_schedule_with_warmup


def create_optimizer_and_scheduler(
    model,
    learning_rate: float,
    num_epochs: int,
    train_dataloader,
    weight_decay: float = 1e-2,
):
    """AdamW + linear warm-up/decay."""
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped, lr=learning_rate)

    steps_per_epoch   = len(train_dataloader)
    num_training_steps = steps_per_epoch * num_epochs
    num_warmup_steps   = int(0.1 * num_training_steps)   # 10 % warm-up

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def run_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    *,
    train: bool,
    grad_clip: float,
    scaler: torch.cuda.amp.GradScaler | None,
    top_k: int = 1,
) -> Dict[str, float]:
    """One full pass over `dataloader`."""
    model.train() if train else model.eval()

    total_loss, n_samples = 0.0, 0
    preds_all, labels_all, logits_all = [], [], []

    loop = tqdm(dataloader, desc="train" if train else "eval", leave=False)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(device_type="cuda", dtype=dtype, enabled=scaler is not None):
            out    = model(**batch)
            loss   = out.loss
            logits = out.logits                            # [B, C]

        if train:
            (scaler.scale(loss) if scaler else loss).backward()

            if grad_clip is not None:
                if scaler:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        # ── bookkeeping ───────────────────────────────────
        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        n_samples  += bs

        preds_all .extend(logits.argmax(dim=-1).cpu())
        labels_all.extend(batch["labels"].cpu())
        logits_all.append(logits.detach().cpu().float())   # keep on CPU, fp32

    # ── metrics ──────────────────────────────────────────
    acc1 = accuracy_score(labels_all, preds_all)

    logits_tensor = torch.vstack(logits_all)               # [N, C]
    num_classes   = logits_tensor.shape[1]

    acck = (
        top_k_accuracy_score(
            labels_all,
            logits_tensor.numpy(),
            k=top_k,
            labels=list(range(num_classes)),               # <- explicit C classes
        )
        if top_k > 1 else acc1
    )

    f1  = f1_score(labels_all, preds_all, average="weighted")

    return {
        "loss": total_loss / n_samples,
        "acc1": acc1,
        "acck": acck,
        "f1":   f1,
    }


def train_model(
    model,
    train_dl,
    val_dl,
    device,
    *,
    num_epochs: int,
    learning_rate: float,
    checkpoint_dir: Path,
    grad_clip: float = 1.0,
    use_amp: bool = True,
    logger: Optional[object] = None,      # e.g. WandBLogger
    top_k: int = 1,
) -> Dict[str, List[float]]:
    """High-level loop that returns a history dict."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model, learning_rate, num_epochs, train_dl
    )
    scaler = torch.amp.GradScaler(device) if (use_amp and torch.cuda.is_available()) else None

    history = {
    "train_loss": [],
    "val_loss":   [],
    "val_acc1":   [],
    "val_acck":   [],
    "val_f1":     [],
    }

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n— Epoch {epoch}/{num_epochs} —")

        train_stats = run_epoch(
            model, train_dl, optimizer, scheduler, device,
            train=True, grad_clip=grad_clip, scaler=scaler,
            top_k=top_k,
        )
        val_stats = run_epoch(
            model, val_dl, optimizer, scheduler, device,
            train=False, grad_clip=grad_clip, scaler=None,
            top_k=top_k,
        )

        # console log
        print(
            f"train loss {train_stats['loss']:.4f} | "
            f"val loss {val_stats['loss']:.4f} | "
            f"val F1 {val_stats['f1']:.4f} | "
            f"val 1 acc {val_stats['acc1']:.4f} | "
            f"val {top_k} acc {val_stats['acck']:.4f}"
        )

        if logger:
            logger.log({
                "epoch": epoch,
                "train/loss": train_stats["loss"],
                "val/loss":   val_stats["loss"],
                "val/f1":     val_stats["f1"],
                "val/1_acc":  val_stats["acc1"],
                f"val/{top_k}_acc": val_stats["acck"],
            }, step=epoch)

        # checkpoint on best val-F1
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

        # store metrics in memory for plotting
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc1"].append(val_stats["acc1"])
        history["val_acck"].append(val_stats["acck"])
        history["val_f1"].append(val_stats["f1"])

    if logger:
        logger.log_artifact(
            checkpoint_dir / "best.pt",
            name="best_model",
            type_="model",
        )
        logger.finish()

    return history
