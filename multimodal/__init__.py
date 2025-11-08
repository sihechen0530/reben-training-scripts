"""
Multimodal classification module for combining S1 and S2 satellite data.

This module provides a flexible architecture for multimodal classification:
- DINOv3 backbone for S2 RGB data (3 channels)
- ResNet101 backbone for S2 non-RGB channels (11 channels) + S1 data (2 channels)
- Configurable late fusion strategies
- Flexible classification heads
"""

from multimodal.model import MultiModalModel

__all__ = ['MultiModalModel']

