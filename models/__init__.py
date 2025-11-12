"""
Models module for BigEarthNet v2.0 training.

This module provides:
- Unimodal classifiers: ResNet, DINOv3 + linear probe, etc.
- Multimodal classifiers: Late fusion with concatenation and linear classifier
- Backbone models: Factory functions for creating backbones
"""

from models.base import BaseClassifier, BaseUnimodalClassifier, BaseMultimodalClassifier
from models.fusion import ConcatenationFusion, FusionStrategy
from models.heads import LinearClassifierHead, ClassifierHead
from models.backbone_config import BackboneConfig

__all__ = [
    "BaseClassifier",
    "BaseUnimodalClassifier",
    "BaseMultimodalClassifier",
    "ConcatenationFusion",
    "FusionStrategy",
    "LinearClassifierHead",
    "ClassifierHead",
    "BackboneConfig",
]

# Import backbone factory
try:
    from models.backbones import create_backbone_from_config
    __all__.append("create_backbone_from_config")
except ImportError:
    pass

# Import unimodal classifiers
try:
    from models.unimodal import ResNetClassifier, DINOv3LinearProbeClassifier
    __all__.extend([
        "ResNetClassifier",
        "DINOv3LinearProbeClassifier",
    ])
except ImportError:
    pass

# Import multimodal classifiers
try:
    from models.multimodal import MultimodalLateFusionClassifier
    __all__.extend([
        "MultimodalLateFusionClassifier",
    ])
except ImportError:
    pass

