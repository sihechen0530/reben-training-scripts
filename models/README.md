# Classifiers Module

This module provides a clean separation between unimodal and multimodal classifiers for BigEarthNet v2.0 training.

## Structure

```
classifiers/
├── __init__.py           # Module exports
├── base.py              # Base classes for classifiers
├── backbone_config.py   # Backbone configuration dataclass
├── fusion.py            # Fusion strategies (concatenation, weighted sum, etc.)
├── heads.py             # Classification heads (linear, MLP, etc.)
├── unimodal.py          # Unimodal classifiers (ResNet, DINOv3 + linear probe)
└── multimodal.py        # Multimodal classifiers (late fusion)
```

## Unimodal Classifiers

Unimodal classifiers use a single backbone with a classification head.

### ResNetClassifier

ResNet-based classifier using ConfigILM/timm models.

```python
from models import ResNetClassifier

classifier = ResNetClassifier(
    architecture="resnet18",
    num_classes=19,
    num_channels=10,
    image_size=120,
    drop_rate=0.15,
)
```

### DINOv3LinearProbeClassifier

DINOv3 backbone with linear probe classifier. Supports freezing the backbone for linear probing.

```python
from models import DINOv3LinearProbeClassifier

classifier = DINOv3LinearProbeClassifier(
    model_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    num_classes=19,
    num_input_channels=3,
    image_size=120,
    freeze_backbone=True,  # Linear probing mode
)
```

## Multimodal Classifiers

Multimodal classifiers fuse features from multiple backbones with different input channels.

### MultimodalLateFusionClassifier

Late fusion classifier that concatenates embeddings from different backbones and uses a linear classifier (default implementation).

**Default behavior:**
- Fusion: Concatenation of embeddings
- Classifier: Linear layer with LayerNorm and Dropout

```python
from models import MultimodalLateFusionClassifier, BackboneConfig

# Create backbone configurations
backbone_configs = [
    BackboneConfig(name="resnet18_3ch", architecture="resnet18", input_channels=3),
    BackboneConfig(name="resnet50_10ch", architecture="resnet50", input_channels=10),
]

# Create multimodal classifier
classifier = MultimodalLateFusionClassifier(
    backbone_configs=backbone_configs,
    image_size=120,
    num_classes=19,
    drop_rate=0.15,
    freeze_backbones=False,  # Set to True to train only classifier
)
```

## Customization

### Custom Fusion Strategies

You can implement custom fusion strategies by extending `FusionStrategy`:

```python
from models.fusion import FusionStrategy
import torch

class CustomFusion(FusionStrategy):
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # Your fusion logic here
        return fused_features
```

### Custom Classification Heads

You can implement custom classification heads by extending `ClassifierHead`:

```python
from models.heads import ClassifierHead
import torch.nn as nn

class CustomClassifierHead(ClassifierHead):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes),
            # Your custom layers
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)
```

## Usage Examples

### Training Unimodal Classifier

```python
from models import ResNetClassifier

classifier = ResNetClassifier(
    architecture="resnet50",
    num_channels=10,
    num_classes=19,
)
```

### Training Multimodal Classifier

```python
from models import MultimodalLateFusionClassifier, BackboneConfig

backbone_configs = [
    BackboneConfig(name="rgb", architecture="resnet18", input_channels=3),
    BackboneConfig(name="multispectral", architecture="resnet50", input_channels=10),
]

classifier = MultimodalLateFusionClassifier(
    backbone_configs=backbone_configs,
    freeze_backbones=True,  # Train only classifier
)
```

### Using Custom Fusion and Classifier

```python
from models import MultimodalLateFusionClassifier, BackboneConfig
from models.fusion import WeightedSumFusion
from models.heads import MLPClassifierHead

backbone_configs = [
    BackboneConfig(name="backbone1", architecture="resnet18", input_channels=3),
    BackboneConfig(name="backbone2", architecture="resnet50", input_channels=10),
]

# Custom fusion
fusion = WeightedSumFusion(
    feature_dims=[512, 2048],  # Feature dims from backbones
    learnable_weights=True,
)

# Custom classifier
classifier_head = MLPClassifierHead(
    input_dim=2048,  # Output dim from fusion
    num_classes=19,
    hidden_dims=[1024, 512],
)

classifier = MultimodalLateFusionClassifier(
    backbone_configs=backbone_configs,
    fusion=fusion,
    classifier=classifier_head,
)
```

## Migration from Old Code

The old `MultiBackboneClassifier` in `reben_publication` has been replaced by `MultimodalLateFusionClassifier`. The new implementation:

1. **Separates concerns**: Fusion and classifier are separate modules
2. **More flexible**: Easier to customize fusion strategies and classification heads
3. **Better structure**: Clear base classes for unimodal and multimodal classifiers
4. **Default behavior**: Concatenation fusion + linear classifier (same as before)

To migrate:
- Replace `MultiBackboneClassifier` with `MultimodalLateFusionClassifier`
- Replace `BackboneConfig` import from `reben_publication` to `models`
- The API is largely the same, but now you can optionally customize fusion and classifier

