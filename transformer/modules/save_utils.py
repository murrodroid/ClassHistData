from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json, yaml, torch, matplotlib.pyplot as plt

def _time_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_run_dir(root: str | Path = "runs") -> Path:
    """
    Makes  runs/20250501_093422/  and returns the Path.
    """
    run_dir = Path(__file__).resolve().parent.parent / root / _time_stamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# ------------------ checkpoint helpers ------------------------

def save_best_checkpoint(
    run_dir: Path,
    state: dict,
    filename: str = "best.pt",
):
    path = run_dir / filename
    torch.save(state, path)
    print(f"✓ Best checkpoint saved to {path}")


def export_hf_format(
    run_dir: Path,
    model,
    tokenizer,
    subdir: str = "model",
):
    """
    Saves exactly the structure that HuggingFace looks for:
    config.json, model.safetensors, tokenizer.json, …
    """
    tgt = run_dir / subdir
    model.save_pretrained(tgt)
    tokenizer.save_pretrained(tgt)
    print(f"✓ HF-compatible model saved at {tgt}")


# ------------------ metrics & history -------------------------

def save_metrics(run_dir: Path, metrics: dict, fname: str = "metrics.json"):
    path = run_dir / fname
    with open(path, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"✓ Metrics written to {path}")

def save_history(history: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Enrich with a timestamp – handy when you aggregate runs later
    stamped = {
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **history,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(stamped, f, indent=2)

def load_history(path: Path) -> dict:
    """Utility for later plotting or comparison."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)

def plot_history(run_dir: Path, history: dict):
    """
    Generates train/val loss curve PNG for quick glance.
    """
    plt.figure()
    plt.plot(history["train_loss"], label="train-loss")
    plt.plot(history["val_loss"],   label="val-loss")
    plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss")
    png = run_dir / "loss_curve.png"
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Loss curve stored at {png}")


# ------------------ config snapshot ---------------------------

def dump_config(run_dir: Path, cfg: dict, fname: str = "config.yaml"):
    path = run_dir / fname
    with open(path, "w") as fp:
        yaml.safe_dump(cfg, fp)
    print(f"✓ Config dumped to {path}")
