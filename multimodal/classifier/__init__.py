"""
Classification heads for multimodal models.

This module provides classification head implementations:
- LinearClassifierHead: Simple linear classifier (default)
- MLPClassifierHead: Multi-layer perceptron classifier
"""

from multimodal.classifier.classification_head import (
    ClassifierHead,
    LinearClassifierHead,
    MLPClassifierHead,
)

__all__ = ['ClassifierHead', 'LinearClassifierHead', 'MLPClassifierHead']

