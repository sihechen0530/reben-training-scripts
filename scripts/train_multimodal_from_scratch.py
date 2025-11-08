"""
Training script for multimodal model with DINOv3 and ResNet both trained from scratch.

This script provides a convenient way to train the multimodal model where both
DINOv3 and ResNet backbones are trained from scratch (no pretrained weights).

Usage:
    python train_multimodal_from_scratch.py
"""
import sys
import os
from pathlib import Path

# Add parent directory to path to allow importing multimodal module
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Change to scripts directory
os.chdir(script_dir)

# Import the main training function
from train_multimodal import main as train_main
import typer

def main(
    seed: int = 42,
    lr: float = 0.001,
    epochs: int = 100,
    bs: int = 32,
    drop_rate: float = 0.15,
    warmup: int = 1000,
    workers: int = 8,
    use_wandb: bool = False,
    test_run: bool = False,
    dinov3_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    dinov3_lr: float = 1e-4,
    resnet_lr: float = 1e-4,
    fusion_type: str = "concat",
    classifier_type: str = "linear",
):
    """
    Train multimodal model with both DINOv3 and ResNet trained from scratch.
    
    Both backbones will be initialized without pretrained weights and will be
    trained during the training process.
    """
    print("=" * 80)
    print("Training Multimodal Model - Both Backbones from Scratch")
    print("=" * 80)
    print(f"DINOv3: model={dinov3_model_name}, pretrained=False, freeze=False")
    print(f"ResNet: pretrained=False, freeze=False")
    print("=" * 80)
    
    # Call the main training function with from-scratch settings
    train_main(
        seed=seed,
        lr=lr,
        epochs=epochs,
        bs=bs,
        drop_rate=drop_rate,
        warmup=warmup,
        workers=workers,
        use_wandb=use_wandb,
        test_run=test_run,
        dinov3_model_name=dinov3_model_name,
        dinov3_pretrained=False,  # No pretrained weights
        dinov3_freeze=False,      # Allow training
        dinov3_lr=dinov3_lr,
        dinov3_checkpoint=None,
        resnet_pretrained=False,  # No pretrained weights
        resnet_freeze=False,     # Allow training
        resnet_lr=resnet_lr,
        resnet_checkpoint=None,
        fusion_type=fusion_type,
        fusion_output_dim=None,
        classifier_type=classifier_type,
        classifier_hidden_dim=512,
        resume_from=None,
    )


if __name__ == "__main__":
    typer.run(main)

