"""
Fusion modules for combining features from multiple backbones.

This module provides late fusion strategies:
- ConcatFusion: Simple concatenation
- WeightedSumFusion: Weighted sum with learnable or fixed weights
- LinearProjectionFusion: Learnable linear projection + sum
"""

from multimodal.fusion.fusion_module import (
    LateFusion,
    ConcatFusion,
    WeightedSumFusion,
    LinearProjectionFusion,
)

__all__ = [
    'LateFusion',
    'ConcatFusion',
    'WeightedSumFusion',
    'LinearProjectionFusion',
]

