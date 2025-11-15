"""
Fusion modules for combining features from multiple backbones.

This module provides late fusion strategies:
- ConcatFusion: Simple concatenation
- WeightedSumFusion: Weighted sum with learnable or fixed weights
- LinearProjectionFusion: Learnable linear projection + sum
- AttentionFusion: Attention-based fusion with learnable queries
- BilinearFusion: Bilinear pooling fusion for fine-grained interaction
- GatedFusion: Gated fusion with learnable gates
"""

from multimodal.fusion.fusion_module import (
    LateFusion,
    ConcatFusion,
    WeightedSumFusion,
    LinearProjectionFusion,
    AttentionFusion,
    BilinearFusion,
    GatedFusion,
)

__all__ = [
    'LateFusion',
    'ConcatFusion',
    'WeightedSumFusion',
    'LinearProjectionFusion',
    'AttentionFusion',
    'BilinearFusion',
    'GatedFusion',
]

