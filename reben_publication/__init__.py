"""
reben_publication package for BigEarthNet v2.0 training scripts.
"""

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from reben_publication.DINOv3Backbone import DINOv3Backbone

__all__ = [
    "BigEarthNetv2_0_ImageClassifier",
    "DINOv3Backbone",
]

try:
    from reben_publication.MultiBackboneClassifier import (
        MultiBackboneClassifier,
        MultiBackboneFeatureExtractor,
        BackboneConfig,
    )
    __all__.extend([
        "MultiBackboneClassifier",
        "MultiBackboneFeatureExtractor",
        "BackboneConfig",
    ])
except ImportError:
    pass

