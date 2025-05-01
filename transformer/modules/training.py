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
from sklearn.metrics        import accuracy_score, f1_score
from transformers           import get_linear_schedule_with_warmup

# ---------------------------------------------------------------
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


# ---------------------------------------------------------------
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
) -> Dict[str, float]:
    """One full pass over `dataloader`."""
    model.train() if train else model.eval()

    total_loss, n_samples = 0.0, 0
    preds_all, labels_all = [], []

    loop = tqdm(dataloader, desc="train" if train else "eval", leave=False)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(device_type="cuda", dtype=dtype, enabled=scaler is not None):
            out    = model(**batch)
            loss   = out.loss
            logits = out.logits

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

        # bookkeeping
        bs = batch["labels"].size(0)
        total_loss += loss.item() * bs
        n_samples  += bs
        preds_all.extend(logits.argmax(dim=-1).detach().cpu())
        labels_all.extend(batch["labels"].detach().cpu())

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="weighted")

    return {"loss": total_loss / n_samples, "acc": acc, "f1": f1}


# ---------------------------------------------------------------
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
) -> Dict[str, List[float]]:
    """High-level loop that returns a history dict."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer, scheduler = create_optimizer_and_scheduler(
        model, learning_rate, num_epochs, train_dl
    )
    scaler = torch.amp.GradScaler(device) if (use_amp and torch.cuda.is_available()) else None


    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n— Epoch {epoch}/{num_epochs} —")

        train_stats = run_epoch(
            model, train_dl, optimizer, scheduler, device,
            train=True, grad_clip=grad_clip, scaler=scaler
        )
        val_stats = run_epoch(
            model, val_dl, optimizer, scheduler, device,
            train=False, grad_clip=grad_clip, scaler=None
        )

        # console log
        print(
            f"train loss {train_stats['loss']:.4f} | "
            f"val loss {val_stats['loss']:.4f} | "
            f"val F1 {val_stats['f1']:.4f}"
        )

        # wandb / external logger
        if logger:
            logger.log({
                "epoch": epoch,
                "train/loss": train_stats["loss"],
                "val/loss":   val_stats["loss"],
                "val/acc":    val_stats["acc"],
                "val/f1":     val_stats["f1"],
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

            if logger:
                logger.log_artifact(
                    checkpoint_dir / "best.pt",
                    name="best_model",
                    type_="model",
                )

        # store metrics in memory for plotting
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])
        history["val_f1"].append(val_stats["f1"])

    if logger:
        logger.finish()

    return history