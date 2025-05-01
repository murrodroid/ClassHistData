# modules/config.py
from pathlib import Path
from datetime import datetime

# ⚙️  core experiment parameters ------------------------------------
model_name   = "emilyalsentzer/Bio_ClinicalBERT"
target       = "icd10h_category"

hyperparams = dict(
    learning_rate = 3e-5,
    batch_size    = 16,
    num_epochs    = 64,
    dropout_rate  = 0.65,
    max_length    = 256,
    top_k         = 3,
)

# 📁  output locations ----------------------------------------------
model_tag = model_name.rsplit("/", 1)[-1]
run_name  = f"{model_tag}_{datetime.now():%Y%m%d_%H%M%S}"
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