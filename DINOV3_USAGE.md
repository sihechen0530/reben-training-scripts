# Using DINOv3 as a Backbone in reben-training-scripts

This guide explains how to use DINOv3 models as backbones in the BigEarthNet v2.0 training scripts.

## Installation

First, ensure you have the required dependencies:

```bash
pip install transformers>=4.40.0
```

Or if using Poetry:

```bash
poetry install
```

## Usage

### Basic Usage

To train with DINOv3, simply specify a DINOv3 architecture name and band configuration:

```bash
python scripts/train_BigEarthNetv2_0.py --architecture=dinov3-base --bandconfig=s2 --no-test-run
```

**Note:** DINOv3 is particularly well-suited for RGB training (`--bandconfig=rgb`) since it was pretrained on RGB images. RGB uses 3 channels (B04, B03, B02) from Sentinel-2.

### Available DINOv3 Models

The following DINOv3 model variants are supported:

- `dinov3-small` or `dinov3-s` - Uses `facebook/dinov3-vits16-pretrain-lvd1689m` (384 dim, default)
- `dinov3-base` or `dinov3-b` - Uses `facebook/dinov3-vitb16-pretrain-lvd1689m` (768 dim)
- `dinov3-large` or `dinov3-l` - Uses `facebook/dinov3-vitl16-pretrain-lvd1689m` (1024 dim)
- `dinov3-giant` or `dinov3-g` - Uses `facebook/dinov3-vit7b16-pretrain-lvd1689m` (1536 dim)

### Specifying Custom DINOv3 Model

You can also specify a custom HuggingFace model name:

```bash
python scripts/train_BigEarthNetv2_0.py \
    --architecture=dinov3-base \
    --dinov3-model-name=facebook/dinov3-vitb16-pretrain-lvd1689m \
    --bandconfig=s2 \
    --no-test-run
```

**Note:** Some DINOv3 models may be gated and require authentication. If you encounter access errors:
1. Log in to HuggingFace: `huggingface-cli login`
2. Request access to the model on the HuggingFace website if needed

### Example Commands

**Train with DINOv3-small (default) on Sentinel-2 bands:**
```bash
python scripts/train_BigEarthNetv2_0.py \
    --architecture=dinov3-small \
    --bandconfig=s2 \
    --bs=32 \
    --lr=0.001 \
    --epochs=100 \
    --no-test-run \
    --use-wandb
```

**Train with DINOv3-base on Sentinel-2 bands:**
```bash
python scripts/train_BigEarthNetv2_0.py \
    --architecture=dinov3-base \
    --bandconfig=s2 \
    --bs=16 \
    --lr=0.0005 \
    --epochs=100 \
    --no-test-run \
    --use-wandb
```

**Train with DINOv3-base on RGB bands (3 channels):**
```bash
python scripts/train_BigEarthNetv2_0.py \
    --architecture=dinov3-base \
    --bandconfig=rgb \
    --bs=32 \
    --lr=0.001 \
    --epochs=100 \
    --no-test-run \
    --use-wandb
```

**Train with DINOv3-large on all bands:**
```bash
python scripts/train_BigEarthNetv2_0.py \
    --architecture=dinov3-large \
    --bandconfig=all \
    --bs=8 \
    --lr=0.0005 \
    --epochs=100 \
    --no-test-run \
    --use-wandb
```

## Testing Trained Models

After training, you can test your DINOv3 checkpoint using the `test_checkpoint_BigEarthNetv2_0.py` script. The script automatically extracts model configuration from the checkpoint, so you typically only need to specify the checkpoint path.

### Basic Testing

The simplest way to test a checkpoint is to just provide the checkpoint path. The script will automatically infer the architecture and band configuration from the checkpoint metadata or filename:

```bash
python scripts/test_checkpoint_BigEarthNetv2_0.py \
    --checkpoint-path ./checkpoints/dinov3-base-42-10-val_mAP_macro-0.85.ckpt
```

### Testing with Explicit Parameters

If auto-detection fails or you want to override parameters, you can specify them explicitly:

```bash
python scripts/test_checkpoint_BigEarthNetv2_0.py \
    --checkpoint-path ./checkpoints/dinov3-base-42-10-val_mAP_macro-0.85.ckpt \
    --architecture dinov3-base \
    --bandconfig s2 \
    --bs 32 \
    --workers 8
```

### Testing Example Commands

**Test a DINOv3-small checkpoint on Sentinel-2 bands:**
```bash
python scripts/test_checkpoint_BigEarthNetv2_0.py \
    --checkpoint-path ./checkpoints/dinov3-small-42-10-val_mAP_macro-0.82.ckpt \
    --architecture dinov3-small \
    --bandconfig s2
```

**Test a DINOv3-base checkpoint on all bands:**
```bash
python scripts/test_checkpoint_BigEarthNetv2_0.py \
    --checkpoint-path ./checkpoints/dinov3-base-42-12-val_mAP_macro-0.88.ckpt \
    --architecture dinov3-base \
    --bandconfig all \
    --bs 16
```

**Quick test with limited batches (for debugging):**
```bash
python scripts/test_checkpoint_BigEarthNetv2_0.py \
    --checkpoint-path ./checkpoints/dinov3-base-42-10-val_mAP_macro-0.85.ckpt \
    --test-run
```

### Understanding Checkpoint Filenames

Checkpoints are saved with the following naming convention:
```
{architecture}-{seed}-{channels}-val_mAP_macro-{val_mAP:.2f}.ckpt
```

For example:
- `dinov3-base-42-10-val_mAP_macro-0.85.ckpt` = DINOv3-base, seed 42, 10 channels (S2), validation mAP 0.85
- `dinov3-large-42-12-val_mAP_macro-0.91.ckpt` = DINOv3-large, seed 42, 12 channels (all bands), validation mAP 0.91

The script automatically extracts architecture and band configuration from this filename if checkpoint metadata is not available.

### Test Output

The script will output comprehensive test metrics including:
- Test loss
- Multilabel Average Precision (macro and micro)
- Multilabel F1 Score (macro and micro)
- Multilabel Precision (macro and micro)
- Class-wise accuracy for all 19 BigEarthNet v2.0 classes

Example output:
```
============================================================
Testing checkpoint: ./checkpoints/dinov3-base-42-10-val_mAP_macro-0.85.ckpt
Architecture: dinov3-base
Band config: s2 (10 channels)
============================================================

============================================================
Test Results:
============================================================
test/loss: 0.234567
test/MultilabelAveragePrecision_macro: 0.850123
test/MultilabelAveragePrecision_micro: 0.867890
test/MultilabelF1Score_macro: 0.782345
test/MultilabelF1Score_micro: 0.801234
...
============================================================
```

## Implementation Details

### Architecture Adaptation

DINOv3 models are designed for RGB (3-channel) inputs. When using multi-channel inputs (e.g., 10 bands for Sentinel-2, 12 bands for Sentinel-2+Sentinel-1), the model automatically adapts the input layer:

- For inputs with more than 3 channels: The RGB weights are repeated/replicated to handle additional channels
- For inputs with fewer than 3 channels: Only the first N channels are used

### Integration

The DINOv3 integration works seamlessly with the existing training pipeline:

1. The `DINOv3Backbone` class wraps the HuggingFace transformers DINOv3 model
2. It's automatically detected when architecture names start with "dinov3"
3. All existing training parameters (learning rate, dropout, etc.) work the same way
4. The model is compatible with the HuggingFace Hub upload functionality

### Differences from Timm Models

- DINOv3 uses Vision Transformer architecture (attention-based)
- Input normalization may differ from standard timm models
- Feature extraction uses CLS token from the transformer output

## Troubleshooting

### Import Errors

If you see an error about transformers not being available:

```bash
pip install transformers>=4.40.0
```

### Model Loading Issues

If DINOv3 models fail to load, check:

1. Internet connection (models are downloaded from HuggingFace)
2. HuggingFace Hub login: `huggingface-cli login`
3. Model name is correct (use `facebook/dinov3-*` format)

### Multi-Channel Input Issues

If you encounter errors with multi-channel inputs:

- The automatic adaptation should work for standard cases
- For custom channel configurations, you may need to modify `DINOv3Backbone._adapt_input_layer()`

### Testing Issues

If you encounter errors when testing checkpoints:

1. **Configuration mismatch**: Ensure the architecture and bandconfig parameters match the training configuration. The script tries to auto-detect these, but you may need to specify them explicitly:
   ```bash
   python scripts/test_checkpoint_BigEarthNetv2_0.py \
       --checkpoint-path ./checkpoints/your-checkpoint.ckpt \
       --architecture dinov3-base \
       --bandconfig s2
   ```

2. **DINOv3 model loading**: When testing, the script needs to recreate the model architecture. If you used a custom `dinov3-model-name` during training, you may need to specify it during testing:
   ```bash
   python scripts/test_checkpoint_BigEarthNetv2_0.py \
       --checkpoint-path ./checkpoints/your-checkpoint.ckpt \
       --architecture dinov3-base \
       --dinov3-model-name facebook/dinov3-vitb16-pretrain-lvd1689m
   ```

3. **Checkpoint not found**: Ensure you're running from the `scripts` directory and the checkpoint path is correct:
   ```bash
   cd scripts
   python test_checkpoint_BigEarthNetv2_0.py --checkpoint-path ../checkpoints/your-checkpoint.ckpt
   ```

## Notes

- DINOv3 models are larger than typical ResNet models, so you may need to reduce batch size
- Learning rates may need adjustment (typically start lower, e.g., 1e-4 to 1e-3)
- DINOv3 models require more GPU memory than ResNet models

