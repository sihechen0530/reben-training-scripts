# Multimodal Classification Module

A flexible PyTorch module for multimodal classification of S1 and S2 satellite data.

## Architecture Overview

This module implements a multimodal architecture that:

1. **S2 RGB data (3 channels)** → DINOv3 backbone (pretrained or trainable)
2. **S2 non-RGB channels (11 channels) + S1 data (2 channels)** → ResNet101 backbone (pretrained or trainable)
3. **Late Fusion** of embeddings from both backbones
4. **Classification head** for final predictions

## Directory Structure

```
multimodal/
├── __init__.py
├── backbones/
│   ├── __init__.py
│   ├── dinov3_backbone.py      # DINOv3 backbone for RGB
│   └── resnet_backbone.py      # ResNet101 backbone for multi-channel data
├── fusion/
│   ├── __init__.py
│   └── fusion_module.py        # Late fusion strategies
├── classifier/
│   ├── __init__.py
│   └── classification_head.py  # Classification heads
├── model.py                    # Main MultiModalModel class
└── README.md                   # This file
```

## Quick Start

### Basic Usage

```python
import torch
from multimodal import MultiModalModel

# Create model with default configuration
model = MultiModalModel()

# Example input data
batch_size = 4
image_size = 120
s1_data = torch.randn(batch_size, 2, image_size, image_size)  # S1: 2 channels
s2_data = torch.randn(batch_size, 14, image_size, image_size)  # S2: 14 channels (3 RGB + 11 non-RGB)

# Forward pass
logits = model(s1_data, s2_data)
print(f"Logits shape: {logits.shape}")  # (batch_size, num_classes)

# Get embeddings as well
logits, embeddings = model(s1_data, s2_data, return_embeddings=True)
print(f"DINOv3 embeddings: {embeddings['dinov3'].shape}")
print(f"ResNet embeddings: {embeddings['resnet'].shape}")
print(f"Fused embeddings: {embeddings['fused'].shape}")
```

### Custom Configuration

```python
config = {
    "backbones": {
        "dinov3": {
            "model_name": "facebook/dinov3-vitb16-pretrain-lvd1689m",  # Base model
            "pretrained": True,
            "freeze": False,  # Fine-tune DINOv3
            "lr": 1e-4,  # Lower learning rate for pretrained backbone
        },
        "resnet101": {
            "pretrained": True,
            "freeze": False,  # Fine-tune ResNet101
            "lr": 1e-4,
        },
    },
    "fusion": {
        "type": "weighted",  # Options: "concat", "weighted", "linear_projection"
    },
    "classifier": {
        "type": "mlp",  # Options: "linear", "mlp"
        "hidden_dim": 512,
        "num_classes": 19,
        "drop_rate": 0.15,
    },
    "image_size": 120,
}

model = MultiModalModel(config=config)
```

### Training with Different Learning Rates

```python
import torch.optim as optim

model = MultiModalModel(config=config)

# Get parameter groups with different learning rates
param_groups = model.get_parameter_groups()

# Create optimizer with parameter groups
optimizer = optim.AdamW(param_groups)

# Or manually specify learning rates
optimizer = optim.AdamW([
    {"params": model.dinov3_backbone.parameters(), "lr": 1e-4},
    {"params": model.resnet_backbone.parameters(), "lr": 1e-4},
    {"params": model.fusion.parameters(), "lr": 1e-3},
    {"params": model.classifier.parameters(), "lr": 1e-3},
])
```

## Configuration Options

### Backbone Configuration

#### DINOv3 Backbone
- `model_name`: HuggingFace model name
  - `"facebook/dinov3-vits16-pretrain-lvd1689m"` (small, 384 dim)
  - `"facebook/dinov3-vitb16-pretrain-lvd1689m"` (base, 768 dim)
  - `"facebook/dinov3-vitl16-pretrain-lvd1689m"` (large, 1024 dim)
  - `"facebook/dinov3-vitg16-pretrain-lvd1689m"` (giant, 1536 dim)
- `pretrained`: Whether to load pretrained weights (default: True)
- `freeze`: Whether to freeze backbone parameters (default: False)
- `lr`: Learning rate for this backbone (default: 1e-4)

#### ResNet101 Backbone
- `pretrained`: Whether to load pretrained ImageNet weights (default: True)
- `freeze`: Whether to freeze backbone parameters (default: False)
- `lr`: Learning rate for this backbone (default: 1e-4)

### Fusion Configuration

#### Concatenation (`"concat"`)
Simple concatenation of features:
```python
"fusion": {
    "type": "concat",
}
```
Output dimension = sum of input dimensions

#### Weighted Sum (`"weighted"`)
Weighted sum with learnable weights:
```python
"fusion": {
    "type": "weighted",
}
```
Output dimension = max of input dimensions

#### Linear Projection (`"linear_projection"`)
Learnable linear projection followed by sum:
```python
"fusion": {
    "type": "linear_projection",
    "output_dim": 512,  # Optional, defaults to max(input_dims)
}
```

### Classifier Configuration

#### Linear Classifier (`"linear"`)
```python
"classifier": {
    "type": "linear",
    "num_classes": 19,
    "drop_rate": 0.15,
}
```

#### MLP Classifier (`"mlp"`)
```python
"classifier": {
    "type": "mlp",
    "hidden_dim": 512,
    "num_classes": 19,
    "drop_rate": 0.15,
}
```

## Example Configurations

### Configuration 1: Default (Concatenation + Linear)
```python
config = {
    "backbones": {
        "dinov3": {"pretrained": True, "freeze": False, "lr": 1e-4},
        "resnet101": {"pretrained": True, "freeze": False, "lr": 1e-4},
    },
    "fusion": {"type": "concat"},
    "classifier": {"type": "linear", "num_classes": 19, "drop_rate": 0.15},
    "image_size": 120,
}
```

### Configuration 2: Weighted Fusion + MLP
```python
config = {
    "backbones": {
        "dinov3": {"pretrained": True, "freeze": False, "lr": 1e-4},
        "resnet101": {"pretrained": True, "freeze": False, "lr": 1e-4},
    },
    "fusion": {"type": "weighted"},
    "classifier": {"type": "mlp", "hidden_dim": 512, "num_classes": 19, "drop_rate": 0.15},
    "image_size": 120,
}
```

### Configuration 3: Frozen Backbones (Linear Probing)
```python
config = {
    "backbones": {
        "dinov3": {"pretrained": True, "freeze": True, "lr": 1e-4},
        "resnet101": {"pretrained": True, "freeze": True, "lr": 1e-4},
    },
    "fusion": {"type": "linear_projection", "output_dim": 512},
    "classifier": {"type": "mlp", "hidden_dim": 256, "num_classes": 19, "drop_rate": 0.2},
    "image_size": 120,
}
```

## Integration with PyTorch Lightning

```python
import pytorch_lightning as pl
import torch.nn.functional as F

class MultiModalLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = MultiModalModel(config=config)
        self.config = config
    
    def forward(self, s1_data, s2_data):
        return self.model(s1_data, s2_data)
    
    def training_step(self, batch, batch_idx):
        s1_data, s2_data, labels = batch
        logits = self.model(s1_data, s2_data)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        param_groups = self.model.get_parameter_groups()
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer
```

## Notes

- **Input Format**: 
  - `s1_data`: Shape `(B, 2, H, W)` - S1 radar data (2 channels: VH, VV)
  - `s2_data`: Shape `(B, 14, H, W)` - S2 spectral data (14 channels: 3 RGB + 11 non-RGB)
  
- **Channel Ordering**: The S2 data should have RGB channels first (channels 0-2), followed by non-RGB channels (channels 3-13).

- **Freezing Backbones**: When `freeze=True`, backbone parameters are frozen and won't be updated during training. This is useful for linear probing or feature extraction.

- **Learning Rates**: Different learning rates can be set for each backbone, fusion, and classifier components. Use `get_parameter_groups()` to get parameter groups for the optimizer.

