"""
Example training script for multimodal model using DINOv3-large checkpoint.

This script demonstrates how to train a multimodal model with:
- DINOv3-large backbone loaded from checkpoint (frozen)
- ResNet101 backbone with ImageNet pretrained weights (fine-tuned)
- Late fusion (concat) and linear classifier
"""

import subprocess
import sys
from pathlib import Path

# Checkpoint path
DINOV3_CKPT = "/home/chen.sihe1/documents/cs6140/reben-training-scripts/scripts/checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt"

# Training parameters
SEED = 42
LR = 0.001
EPOCHS = 100
BS = 32
DROP_RATE = 0.15
WARMUP = 1000
WORKERS = 8

# Build command
cmd = [
    "python", "train_multimodal.py",
    "--seed", str(SEED),
    "--lr", str(LR),
    "--epochs", str(EPOCHS),
    "--bs", str(BS),
    "--drop-rate", str(DROP_RATE),
    "--warmup", str(WARMUP),
    "--workers", str(WORKERS),
    "--dinov3-checkpoint", DINOV3_CKPT,
    "--dinov3-freeze",
    "--dinov3-lr", "1e-4",
    "--resnet-pretrained",
    "--resnet-lr", "1e-4",
    "--fusion-type", "concat",
    "--classifier-type", "linear",
    "--use-wandb", "False",
    "--test-run", "False",
]

print("=" * 80)
print("Training Multimodal Model")
print("=" * 80)
print(f"DINOv3 checkpoint: {DINOV3_CKPT}")
print(f"Seed: {SEED}")
print(f"Learning rate: {LR}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BS}")
print("=" * 80)
print("\nCommand:")
print(" ".join(cmd))
print("\n" + "=" * 80)

# Run training
if __name__ == "__main__":
    # Make sure we're in the scripts directory
    script_dir = Path(__file__).parent
    if Path(".").resolve().name != "scripts":
        print("Warning: Please run from scripts directory")
        sys.exit(1)
    
    subprocess.run(cmd)

