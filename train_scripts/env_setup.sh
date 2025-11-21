#!/usr/bin/env bash
# ===== Explorer modules =====
newgrp gophers

module load cuda/12.1.1
module load anaconda3/2024.06

# ===== Conda env =====
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "/home/chen.sihe1/.conda/envs/dinov3/"
fi

export TOKENIZERS_PARALLELISM=false

# ===== Hugging Face Token (required for DINOv3) =====
# Replace YOUR_HF_TOKEN_HERE with your actual token
export HF_TOKEN="YOUR_HF_TOKEN_HERE"