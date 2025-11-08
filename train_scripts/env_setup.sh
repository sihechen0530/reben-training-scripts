#!/usr/bin/env bash
# ===== Explorer modules =====
module load cuda/12.1.1
module load anaconda3/2024.06

# ===== Conda env =====
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "/home/chen.sihe1/.conda/envs/dinov3/"
fi

export TOKENIZERS_PARALLELISM=false