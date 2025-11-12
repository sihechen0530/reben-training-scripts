"""
Backbone models for feature extraction.
"""
from models.backbones.factory import create_backbone_from_config, create_resnet_backbone, create_dinov3_backbone

__all__ = [
    "create_backbone_from_config",
    "create_resnet_backbone",
    "create_dinov3_backbone",
]

# Import DINOv3 backbone if available
try:
    from reben_publication.DINOv3Backbone import DINOv3Backbone
    __all__.append("DINOv3Backbone")
except ImportError:
    DINOv3Backbone = None

