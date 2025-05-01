# modules/config.py
from pathlib import Path
from datetime import datetime

# ⚙️  core experiment parameters ------------------------------------
model_name   = "meta-llama/Llama-3.2-1B"   # ← fixed the double “//”
target       = "icd10h_category"

hyperparams = dict(
    learning_rate = 4e-4,
    batch_size    = 32,
    num_epochs    = 64,
    dropout_rate  = 0.55,
    max_length    = 512,
)

# 📁  output locations ----------------------------------------------
run_name    = f"llama1B_{datetime.now():%Y%m%d_%H%M%S}"
runs_root   = Path("runs")                  # top-level folder
run_dir     = runs_root / run_name          # e.g. runs/llama1B_20250501_140700
ckpt_dir    = run_dir / "checkpoints"
reports_dir = run_dir / "reports"
run_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(exist_ok=True)
reports_dir.mkdir(exist_ok=True)

# 📊  WandB ----------------------------------------------------------
wandb_cfg = dict(
    project = "ICD10hClassification",
    entity  = None,            # set if you’re in an org
    mode    = "online",       # "online" on a login node / “offline” on the compute node
)