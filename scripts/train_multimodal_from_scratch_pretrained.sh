#!/bin/bash
# Training script for multimodal model with DINOv3 and ResNet both fine-tuned
# This script uses pretrained weights but allows both backbones to be trained (not frozen)

# Training parameters
SEED=42
LR=0.001
EPOCHS=100
BS=32
DROP_RATE=0.15
WARMUP=1000
WORKERS=8

# DINOv3 configuration - use pretrained but allow fine-tuning
DINOV3_MODEL_NAME="facebook/dinov3-vitb16-pretrain-lvd1689m"  # Can be changed to small/large/giant
DINOV3_PRETRAINED=true   # Use pretrained weights
DINOV3_FREEZE=false      # Allow fine-tuning (not frozen)
DINOV3_LR=1e-4

# ResNet101 configuration - use pretrained but allow fine-tuning
RESNET_PRETRAINED=true   # Use pretrained ImageNet weights
RESNET_FREEZE=false     # Allow fine-tuning (not frozen)
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
    --dinov3-model-name "$DINOV3_MODEL_NAME" \
    --dinov3-pretrained $DINOV3_PRETRAINED \
    --dinov3-freeze $DINOV3_FREEZE \
    --dinov3-lr $DINOV3_LR \
    --resnet-pretrained $RESNET_PRETRAINED \
    --resnet-freeze $RESNET_FREEZE \
    --resnet-lr $RESNET_LR \
    --fusion-type $FUSION_TYPE \
    --classifier-type $CLASSIFIER_TYPE \
    --use-wandb $USE_WANDB \
    --test-run $TEST_RUN

