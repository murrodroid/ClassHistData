#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -q gpua100                       # queue/partition (check if a GPU queue is required)
#BSUB -W 00:30                      # wall-time hh:mm
#BSUB -n 4                         # CPU cores
#BSUB -R "span[hosts=1]"           # keep all cores on one node
#BSUB -R "select[gpu80gb]"
#BSUB -R "rusage[mem=8GB]"         # 4 GB RAM per core  → 16 GB total
#BSUB -gpu "num=1:mode=exclusive_process"   # ← *add* if you need one A100
#BSUB -u s234805@dtu.dk            # where e-mails go
#BSUB -B                           # e-mail at start   (optional)
#BSUB -N                           # e-mail at end     (optional)
#BSUB -oo logs/%J.out              # stdout  (overwrite)
#BSUB -eo logs/%J.err              # stderr  (overwrite)
# -------------------------------------------------

# stop on first error
set -e

# ---- software environment -----
module load cuda/11.8
source /zhome/0e/9/205681/miniconda3/etc/profile.d/conda.sh
conda activate ALenv

nvidia-smi
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

# ---- run your pipeline ---------
python train.py
