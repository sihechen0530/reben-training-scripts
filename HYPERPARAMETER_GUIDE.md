# Hyperparameter Exploration Guide

This guide provides recommendations for exploring hyperparameters when training models on BigEarthNet v2.0.

## Learning Rate

Learning rate is one of the most critical hyperparameters. The optimal value depends on:
- Model architecture (ResNet vs DINOv3)
- Training stage (full training vs fine-tuning vs linear probing)
- Batch size
- Dataset size

### Recommended Learning Rate Values to Try

#### For Full Training (All Parameters)
```bash
# Standard range
--lr 0.001      # Default (good starting point)
--lr 0.0005     # More conservative
--lr 0.0001     # Very conservative (for large models)
--lr 0.002      # More aggressive
--lr 0.003      # Very aggressive (may need lower batch size)

# Linear scaling with batch size (if batch size differs significantly)
# If default is lr=0.001 with bs=32, then:
--lr 0.002 --bs 64   # Scale by 2x if batch size is 2x
--lr 0.0005 --bs 16  # Scale down if batch size is smaller
```

#### For Linear Probing / Fine-tuning (Frozen Backbone)
```bash
# Higher learning rates are often better for linear probing
--lr 0.01       # Common for linear probing
--lr 0.005      # Alternative
--lr 0.03       # More aggressive (for small datasets)
--linear-probe  # Use this flag for DINOv3
```

#### For DINOv3 Models
```bash
# DINOv3 typically needs lower learning rates
--lr 0.0001     # Conservative
--lr 0.0005     # Moderate
--lr 0.001      # Aggressive (may work for smaller variants)
```

### Learning Rate Search Strategy

1. **Start with a wide range search**:
   ```bash
   # Try one order of magnitude apart
   --lr 0.0001
   --lr 0.001
   --lr 0.01
   ```

2. **Then narrow down**:
   ```bash
   # If 0.001 works best, try nearby values
   --lr 0.0005
   --lr 0.001
   --lr 0.002
   ```

3. **Use different learning rates for different model sizes**:
   ```bash
   # Smaller models can handle higher learning rates
   --architecture resnet18 --lr 0.002
   --architecture resnet34 --lr 0.001
   --architecture resnet50 --lr 0.0005
   ```

## Other Important Hyperparameters

### 1. Batch Size

**Impact**: Affects training stability, memory usage, and effective learning rate.

```bash
# Small batch sizes (if memory constrained)
--bs 16
--bs 8

# Medium batch sizes (balanced)
--bs 32    # Default
--bs 64

# Large batch sizes (if you have GPU memory)
--bs 128
--bs 256

# Note: When changing batch size, consider scaling learning rate
# Linear scaling: lr_new = lr_base * (bs_new / bs_base)
```

**Recommendations**:
- Start with 32 or 64
- Use larger batches for better gradient estimates (if memory allows)
- Smaller batches may help with generalization but require lower learning rates

### 2. Dropout Rate (`drop_rate`)

**Impact**: Regularization to prevent overfitting.

```bash
# Lower dropout (more capacity, risk of overfitting)
--drop-rate 0.0   # No dropout
--drop-rate 0.1   # Light regularization

# Medium dropout
--drop-rate 0.15  # Default
--drop-rate 0.2   # Moderate regularization

# Higher dropout (strong regularization)
--drop-rate 0.3
--drop-rate 0.4   # Very strong regularization
```

**Recommendations**:
- Start with 0.15 (default)
- Increase if you see overfitting (high train accuracy, low val accuracy)
- Decrease if model is underfitting
- For pretrained models, often lower dropout is better (0.0-0.1)

### 3. Drop Path Rate (`drop_path_rate`)

**Impact**: Stochastic depth regularization (similar to dropout but for ResNet blocks).

```bash
# No drop path (default for many models)
--drop-path-rate 0.0

# Light drop path
--drop-path-rate 0.1

# Moderate drop path
--drop-path-rate 0.15  # Default

# Strong drop path
--drop-path-rate 0.2
--drop-path-rate 0.3
```

**Recommendations**:
- Most effective for ResNet variants
- Start with 0.0 or 0.1
- Increase if overfitting
- Less critical than dropout for most cases

### 4. Number of Epochs

**Impact**: Training duration and convergence.

```bash
# Short training (for quick experiments)
--epochs 50

# Standard training
--epochs 100  # Default

# Longer training (may need for large models)
--epochs 200
--epochs 300

# Very long training (for convergence)
--epochs 500
```

**Recommendations**:
- Start with 100 epochs
- Use early stopping (enabled by default) to stop when validation doesn't improve
- For fine-tuning, often 50-100 epochs is sufficient
- For training from scratch, may need 200+ epochs

### 5. Warmup Steps

**Impact**: Gradual learning rate increase at start of training.

```bash
# Short warmup
--warmup 500

# Medium warmup
--warmup 1000   # Default

# Long warmup (for large models/datasets)
--warmup 2000
--warmup 5000

# Automatic calculation
--warmup -1     # Lightning will calculate automatically
```

**Recommendations**:
- Use -1 for automatic calculation (recommended)
- Manual warmup: ~10% of total training steps
- Longer warmup for larger models or when using very high learning rates

### 6. Seed (for reproducibility)

**Impact**: Controls randomness in training.

```bash
--seed 42   # Default
--seed 123
--seed 456
```

**Recommendations**:
- Use different seeds to test robustness
- Common practice: run 3-5 seeds and report mean ± std

## Hyperparameter Search Strategies

### 1. Grid Search (Exhaustive)

Try all combinations of a subset of hyperparameters:

```bash
# Example: Learning rate and batch size
for lr in 0.0001 0.001 0.01; do
  for bs in 32 64 128; do
    python train_BigEarthNetv2_0.py --lr $lr --bs $bs --run-name "lr${lr}_bs${bs}"
  done
done
```

### 2. Random Search (More Efficient)

Randomly sample hyperparameter combinations:

```bash
# Random learning rates and batch sizes
python train_BigEarthNetv2_0.py --lr 0.0008 --bs 48 --drop-rate 0.12 --run-name "random1"
python train_BigEarthNetv2_0.py --lr 0.0015 --bs 64 --drop-rate 0.18 --run-name "random2"
python train_BigEarthNetv2_0.py --lr 0.0003 --bs 40 --drop-rate 0.22 --run-name "random3"
```

### 3. Bayesian Optimization (Recommended)

Use tools like:
- Weights & Biases Sweeps
- Optuna
- Ray Tune

## Model-Specific Recommendations

### ResNet Models

```bash
# ResNet18 (small)
--architecture resnet18 --lr 0.001 --bs 64 --drop-rate 0.15 --epochs 100

# ResNet34 (medium)
--architecture resnet34 --lr 0.0008 --bs 64 --drop-rate 0.15 --epochs 120

# ResNet50 (large)
--architecture resnet50 --lr 0.0005 --bs 32 --drop-rate 0.2 --epochs 150
```

### DINOv3 Models

```bash
# DINOv3 Small
--architecture dinov3-small --lr 0.0005 --bs 32 --drop-rate 0.1 --epochs 100

# DINOv3 Base (linear probing)
--architecture dinov3-base --lr 0.01 --bs 64 --drop-rate 0.1 --linear-probe --epochs 50

# DINOv3 Base (full fine-tuning)
--architecture dinov3-base --lr 0.0001 --bs 32 --drop-rate 0.1 --epochs 100
```

## Multi-Backbone Models

For multimodal classifiers, consider:

```bash
# Different learning rates for backbones vs classifier
# (Note: Currently uses same LR, but you can experiment with freezing backbones)

# Train backbones first
python train_multi_backbone.py --backbones "b1:resnet18:3" "b2:resnet50:10" \
    --lr 0.001 --freeze-backbones False --epochs 100

# Then freeze and train classifier only
python train_multi_backbone.py --backbones "b1:resnet18:3" "b2:resnet50:10" \
    --lr 0.01 --freeze-backbones True --epochs 50
```

## Monitoring and Evaluation

### Key Metrics to Watch

1. **Training loss**: Should decrease steadily
2. **Validation loss**: Should decrease and track training loss
3. **Validation mAP**: Should increase
4. **Learning rate**: Monitor via wandb (use `--use-wandb`)

### Signs of Issues

- **Overfitting**: Train loss << Val loss → Increase dropout, reduce model size
- **Underfitting**: Both losses high, not decreasing → Increase model capacity, lower dropout
- **Learning rate too high**: Loss NaN or very unstable → Lower learning rate
- **Learning rate too low**: Very slow convergence → Increase learning rate

## Example Hyperparameter Sweep

```bash
#!/bin/bash
# Run multiple experiments with different hyperparameters

# Learning rate sweep
for lr in 0.0001 0.0005 0.001 0.002 0.005; do
    python train_BigEarthNetv2_0.py \
        --architecture resnet50 \
        --lr $lr \
        --bs 64 \
        --epochs 100 \
        --use-wandb \
        --run-name "lr_sweep_${lr}"
done

# Batch size sweep
for bs in 16 32 64 128; do
    # Scale LR with batch size
    lr=$(echo "0.001 * $bs / 32" | bc -l)
    python train_BigEarthNetv2_0.py \
        --architecture resnet50 \
        --lr $lr \
        --bs $bs \
        --epochs 100 \
        --use-wandb \
        --run-name "bs_sweep_${bs}"
done

# Dropout sweep
for drop in 0.0 0.1 0.15 0.2 0.3; do
    python train_BigEarthNetv2_0.py \
        --architecture resnet50 \
        --lr 0.001 \
        --bs 64 \
        --drop-rate $drop \
        --epochs 100 \
        --use-wandb \
        --run-name "dropout_sweep_${drop}"
done
```

## Best Practices

1. **Start simple**: Begin with default hyperparameters
2. **Change one thing at a time**: Isolate the effect of each hyperparameter
3. **Use validation set**: Don't tune on test set
4. **Track experiments**: Use wandb (`--use-wandb`) to compare runs
5. **Run multiple seeds**: Get confidence intervals (3-5 seeds)
6. **Early stopping**: Let the model stop when it stops improving (enabled by default)
7. **Save checkpoints**: Use `--run-name` to organize experiments

## Quick Reference Table

| Hyperparameter | Default | Low | High | Notes |
|---------------|---------|-----|------|-------|
| Learning Rate | 0.001 | 0.0001 | 0.01 | Lower for large models |
| Batch Size | 32 | 16 | 256 | Scale LR with BS |
| Dropout | 0.15 | 0.0 | 0.4 | Increase if overfitting |
| Drop Path | 0.15 | 0.0 | 0.3 | For ResNets |
| Epochs | 100 | 50 | 500 | Early stopping helps |
| Warmup | 1000 | 500 | 5000 | Use -1 for auto |

