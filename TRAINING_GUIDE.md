# Training Guide

This guide explains how to use the refactored training scripts for ResNet and multi-backbone models.

## 1. Training ResNet with Custom Number of Channels

Use `scripts/train_resnet.py` to train a ResNet model with a designated number of channels.

### Basic Usage

```bash
# Train ResNet18 on 10 channels (S2 bands)
python scripts/train_resnet.py --architecture resnet18 --num_channels 10 --epochs 50 --lr 0.001

# Train ResNet50 on 3 channels (RGB)
python scripts/train_resnet.py --architecture resnet50 --bandconfig rgb --epochs 100 --bs 64

# Train with custom hyperparameters
python scripts/train_resnet.py --architecture resnet34 --num_channels 12 --epochs 200 --lr 0.0005 --bs 128
```

### Available Arguments

- `--architecture`: ResNet architecture (resnet18, resnet34, resnet50, etc.) [default: resnet18]
- `--num_channels`: Number of input channels. If None, uses bandconfig to determine.
- `--bandconfig`: Band configuration (all, s2, s1, rgb, etc.). Ignored if num_channels is specified. [default: s2]
- `--seed`: Random seed [default: 42]
- `--lr`: Learning rate [default: 0.001]
- `--epochs`: Number of epochs [default: 100]
- `--bs`: Batch size [default: 32]
- `--drop_rate`: Dropout rate [default: 0.15]
- `--drop_path_rate`: Drop path rate [default: 0.0]
- `--warmup`: Warmup steps (set to -1 for automatic calculation) [default: 1000]
- `--workers`: Number of data loader workers [default: 8]
- `--use-wandb`: Use wandb for logging [default: False]
- `--upload-to-hub`: Upload model to Huggingface Hub [default: False]
- `--test-run`: Run training with fewer epochs and batches [default: True]
- `--hf-entity`: Huggingface entity to upload the model to (required if upload-to-hub is True)
- `--resume-from`: Path to checkpoint file to resume training from (can be 'best'/'last' or file path)

## 2. Training Multi-Backbone Model with Concatenated Features

Use `scripts/train_multi_backbone.py` to train a model with multiple backbones that have different input channels. Features from all backbones are concatenated and fed to a linear classifier.

### Basic Usage

```bash
# Train with two ResNet18 backbones (3 channels + 10 channels)
python scripts/train_multi_backbone.py \
    --backbones "resnet18_3ch:resnet18:3" "resnet18_10ch:resnet18:10" \
    --epochs 50 --lr 0.001

# Train only the classifier (freeze backbones)
python scripts/train_multi_backbone.py \
    --backbones "resnet18_3ch:resnet18:3" "resnet50_10ch:resnet50:10" \
    --freeze-backbones --lr 0.01

# Train with custom hyperparameters
python scripts/train_multi_backbone.py \
    --backbones "backbone1:resnet18:3" "backbone2:resnet34:10" \
    --epochs 200 --lr 0.0005 --bs 128
```

### Backbone Format

Backbones are specified in the format: `"name:architecture:channels"`

- `name`: Unique name for the backbone (e.g., "resnet18_3ch")
- `architecture`: Architecture name from timm (e.g., "resnet18", "resnet50") or DINOv3 (e.g., "dinov3-base")
- `channels`: Number of input channels for this backbone

### Available Arguments

- `--backbones`: List of backbones in format "name:architecture:channels" [default: ["resnet18:resnet18:3", "resnet18:resnet18:10"]]
- `--image-size`: Input image size [default: 120]
- `--seed`: Random seed [default: 42]
- `--lr`: Learning rate [default: 0.001]
- `--epochs`: Number of epochs [default: 100]
- `--bs`: Batch size [default: 32]
- `--drop-rate`: Dropout rate [default: 0.15]
- `--drop-path-rate`: Drop path rate [default: 0.0]
- `--warmup`: Warmup steps (set to -1 for automatic calculation) [default: 1000]
- `--workers`: Number of data loader workers [default: 8]
- `--freeze-backbones`: Freeze backbone parameters and only train classifier [default: False]
- `--use-wandb`: Use wandb for logging [default: False]
- `--test-run`: Run training with fewer epochs and batches [default: True]
- `--resume-from`: Path to checkpoint file to resume training from (can be 'best'/'last' or file path)

### Examples

#### Example 1: Two ResNet backbones with different channels
```bash
python scripts/train_multi_backbone.py \
    --backbones "rgb_backbone:resnet18:3" "multispectral_backbone:resnet50:10" \
    --epochs 100 --lr 0.001 --bs 64
```

#### Example 2: Train only the linear classifier (freeze backbones)
```bash
python scripts/train_multi_backbone.py \
    --backbones "backbone1:resnet18:3" "backbone2:resnet18:10" \
    --freeze-backbones --lr 0.01 --epochs 50
```

#### Example 3: Mix ResNet and DINOv3 backbones
```bash
python scripts/train_multi_backbone.py \
    --backbones "resnet_backbone:resnet50:10" "dinov3_backbone:dinov3-base:3" \
    --epochs 100 --lr 0.0005
```

## 3. Changing Hyperparameters from Command Line

All training scripts support comprehensive command-line arguments for hyperparameters:

### Learning Rate
```bash
--lr 0.0001  # Lower learning rate
--lr 0.01    # Higher learning rate
```

### Number of Epochs
```bash
--epochs 50   # Short training
--epochs 200  # Longer training
```

### Batch Size
```bash
--bs 16   # Smaller batch size
--bs 128  # Larger batch size
```

### Dropout
```bash
--drop-rate 0.1   # Lower dropout
--drop-rate 0.3   # Higher dropout
```

### Warmup Steps
```bash
--warmup 500     # Fixed warmup steps
--warmup -1      # Automatic calculation
```

## Notes

- Make sure to run scripts from the `scripts/` directory
- The total number of input channels for multi-backbone models is the sum of all backbone channels
- When using `--freeze-backbones`, only the linear classifier is trained (useful for linear probing)
- Checkpoints are saved in the `checkpoints/` directory
- Use `--resume-from best` or `--resume-from last` to resume from the best/last checkpoint

