"""
Classification heads for multimodal models.

This module provides classification head implementations:
- LinearClassifierHead: Simple linear classifier (default)
- MLPClassifierHead: Multi-layer perceptron classifier
- SVMLikeClassifier: SVM-like classifier with hinge loss support
- LabelWiseBinaryClassifier: Binary classifier for each label (multi-label setting)
"""

from multimodal.classifier.classification_head import (
    ClassifierHead,
    LinearClassifierHead,
    MLPClassifierHead,
    SVMLikeClassifier,
    LabelWiseBinaryClassifier,
)

__all__ = [
    'ClassifierHead',
    'LinearClassifierHead',
    'MLPClassifierHead',
    'SVMLikeClassifier',
    'LabelWiseBinaryClassifier',
]

