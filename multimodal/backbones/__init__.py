"""
Backbone modules for multimodal classification.

This module provides backbone implementations:
- DINOv3Backbone: Vision transformer backbone for RGB data
- ResNetBackbone: ResNet backbone for multi-channel spectral/radar data
"""

from multimodal.backbones.dinov3_backbone import DINOv3Backbone
from multimodal.backbones.resnet_backbone import ResNetBackbone

__all__ = ['DINOv3Backbone', 'ResNetBackbone']

