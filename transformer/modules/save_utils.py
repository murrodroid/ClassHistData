# modules/save_utils.py  ← replace this file
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, yaml, torch, matplotlib.pyplot as plt
from transformers import PreTrainedModel, PreTrainedTokenizer

# ────────────────────────────── misc ───────────────────────────────────────── #

def _time_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(root: str | Path = "runs") -> Path:
    run_dir = Path(__file__).resolve().parent.parent / root / _time_stamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# ───────────────────── checkpoint / model persistence ─────────────────────── #

def save_best_checkpoint(
    run_dir: Path,
    state: dict,
    *,
    model: PreTrainedModel | None = None,
    tokenizer: PreTrainedTokenizer | None = None,
    filename: str = "best.pt",
    hf_subdir: str = "model",
) -> None:
    """
    • Always saves *state* to ``run_dir/checkpoints/<filename>``  
    • **If** a ``model`` *and* ``tokenizer`` are supplied, also stores a full
      Hugging Face bundle that can later be re-loaded with
      ``AutoModelForSequenceClassification.from_pretrained(run_dir/hf_subdir)``.
    """
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(state, ckpt_dir / filename)
    print(f"✓ Best state-dict saved to {ckpt_dir/filename}")

    if model is not None and tokenizer is not None:
        export_hf_format(run_dir, model, tokenizer, subdir=hf_subdir)


def export_hf_format(
    run_dir: Path,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *,
    subdir: str = "model",
    push_to_hub: bool = False,
) -> None:
    """
    Persists the exact folder structure Hugging Face expects:
    ``config.json``, ``model.safetensors`` (or ``pytorch_model.bin``),
    ``tokenizer.json``, etc.

    Set *push_to_hub=True* if you want the model immediately uploaded to the
    account configured in ``huggingface-cli login``.
    """
    tgt = run_dir / subdir
    model.save_pretrained(tgt, safe_serialization=True, push_to_hub=push_to_hub)
    tokenizer.save_pretrained(tgt)
    print(f"✓ HF-compatible model written to {tgt}")


# ──────────────────────────── metrics & history ───────────────────────────── #

def save_metrics(run_dir: Path, metrics: dict, fname: str = "metrics.json"):
    path = run_dir / fname
    path.write_text(json.dumps(metrics, indent=2))
    print(f"✓ Metrics written to {path}")


def save_history(history: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **history}
    path.write_text(json.dumps(stamped, indent=2))


def load_history(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def plot_history(run_dir: Path, history: dict):
    plt.figure()
    plt.plot(history["train_loss"], label="train-loss")
    plt.plot(history["val_loss"],   label="val-loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    png = run_dir / "loss_curve.png"
    plt.savefig(png, dpi=150, bbox_inches="tight"); plt.close()
    print(f"✓ Loss curve stored at {png}")


# ─────────────────────────── config snapshot ─────────────────────────────── #

def dump_config(run_dir: Path, cfg: dict, fname: str = "config.yaml"):
    path = run_dir / fname
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"✓ Config dumped to {path}")
