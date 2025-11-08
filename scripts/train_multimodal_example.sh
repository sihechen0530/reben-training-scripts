#!/bin/bash
# Example training script for multimodal model
# Using DINOv3-large checkpoint trained on RGB data

# Checkpoint path
DINOV3_CKPT="/home/chen.sihe1/documents/cs6140/reben-training-scripts/scripts/checkpoints/dinov3-large-42-3-dinov3-large_rgb_42_20251107_120819-val_mAP_macro-0.79.ckpt"

# Training parameters
SEED=42
LR=0.001
EPOCHS=100
BS=32
DROP_RATE=0.15
WARMUP=1000
WORKERS=8

# DINOv3 configuration (will be auto-inferred from checkpoint as large)
# The script will automatically detect "dinov3-large" in the filename
DINOV3_FREEZE=true  # Freeze the pretrained DINOv3 backbone

# ResNet101 configuration
RESNET_PRETRAINED=true  # Use ImageNet pretrained weights
RESNET_FREEZE=false     # Fine-tune ResNet101
RESNET_LR=1e-4

# Fusion and classifier
FUSION_TYPE="concat"  # Options: concat, weighted, linear_projection
CLASSIFIER_TYPE="linear"  # Options: linear, mlp

# Logging
USE_WANDB=false
TEST_RUN=false  # Set to true for quick testing

# Run training
python train_multimodal.py \
    --seed $SEED \
    --lr $LR \
    --epochs $EPOCHS \
    --bs $BS \
    --drop-rate $DROP_RATE \
    --warmup $WARMUP \
    --workers $WORKERS \
    --dinov3-checkpoint "$DINOV3_CKPT" \
    --dinov3-freeze \
    --dinov3-lr 1e-4 \
    --resnet-pretrained \
    --resnet-lr $RESNET_LR \
    --fusion-type $FUSION_TYPE \
    --classifier-type $CLASSIFIER_TYPE \
    --use-wandb $USE_WANDB \
    --test-run $TEST_RUN

