"""
Classification heads for multimodal models.

This module provides flexible classification head implementations
that can be easily replaced or extended.

Available classifiers:
- LinearClassifierHead: Simple linear classifier
- MLPClassifierHead: Multi-layer perceptron
- SVMLikeClassifier: SVM-like classifier with hinge loss support
- LabelWiseBinaryClassifier: Binary classifier for each label (multi-label setting)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from abc import ABC, abstractmethod


class ClassifierHead(nn.Module, ABC):
    """
    Base class for classification heads.
    """
    
    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            features: Input features of shape (B, D)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        pass


class LinearClassifierHead(ClassifierHead):
    """
    Simple linear classification head (default).
    
    Applies LayerNorm, Dropout, and a linear layer.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 19,
        drop_rate: float = 0.15,
        use_layer_norm: bool = True,
    ):
        """
        Initialize linear classifier head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            drop_rate: Dropout rate
            use_layer_norm: Whether to use LayerNorm before the linear layer
        """
        super().__init__()
        
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through linear classifier.
        
        Args:
            features: Input features of shape (B, D)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.classifier(features)


class MLPClassifierHead(ClassifierHead):
    """
    Multi-layer perceptron classification head.
    
    Applies LayerNorm, followed by multiple linear layers with activation and dropout.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 19,
        hidden_dims: Optional[List[int]] = None,
        drop_rate: float = 0.15,
        use_layer_norm: bool = True,
        activation: str = "relu",
    ):
        """
        Initialize MLP classifier head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions (default: [input_dim])
            drop_rate: Dropout rate
            use_layer_norm: Whether to use LayerNorm before the first layer
            activation: Activation function ('relu', 'gelu', 'tanh')
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim]
        
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        
        # Build MLP
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Don't add activation after last layer
                if drop_rate > 0:
                    layers.append(nn.Dropout(drop_rate))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP classifier.
        
        Args:
            features: Input features of shape (B, D)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.classifier(features)


class SVMLikeClassifier(ClassifierHead):
    """
    SVM-like classifier head.
    
    Implements a linear classifier similar to Support Vector Machine.
    The main difference is that it can use hinge loss during training,
    but outputs raw logits (no special activation needed).
    
    Note: For training with hinge loss, use nn.MultiLabelMarginLoss or
    implement a custom hinge loss function. The forward pass outputs
    standard logits that can be used with any loss function.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 19,
        drop_rate: float = 0.15,
        use_layer_norm: bool = True,
        bias: bool = True,
    ):
        """
        Initialize SVM-like classifier head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            drop_rate: Dropout rate
            use_layer_norm: Whether to use LayerNorm before the linear layer
            bias: Whether to use bias in the linear layer (default: True)
        """
        super().__init__()
        
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        
        # Linear layer without activation (SVM outputs raw scores)
        self.linear = nn.Linear(input_dim, num_classes, bias=bias)
        
        if layers:
            self.preprocessing = nn.Sequential(*layers)
        else:
            self.preprocessing = nn.Identity()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SVM-like classifier.
        
        Args:
            features: Input features of shape (B, D)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.preprocessing(features)
        return self.linear(x)


class LabelWiseBinaryClassifier(ClassifierHead):
    """
    Label-wise binary classifier head.
    
    Treats multi-label classification as independent binary classification
    problems for each label. Each class has its own binary classifier
    (linear layer + sigmoid activation).
    
    This is useful for multi-label classification tasks where each sample
    can have multiple positive labels. The output logits can be used with
    BCEWithLogitsLoss for training.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 19,
        drop_rate: float = 0.15,
        use_layer_norm: bool = True,
        shared_backbone: bool = True,
    ):
        """
        Initialize label-wise binary classifier head.
        
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes (labels)
            drop_rate: Dropout rate
            use_layer_norm: Whether to use LayerNorm before the classifiers
            shared_backbone: If True, shares a common feature extractor before
                           individual binary classifiers. If False, each binary
                           classifier is independent. (default: True)
        """
        super().__init__()
        
        # Shared preprocessing (optional shared feature extraction)
        layers = []
        if use_layer_norm:
            layers.append(nn.LayerNorm(input_dim))
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        
        if layers:
            self.preprocessing = nn.Sequential(*layers)
        else:
            self.preprocessing = nn.Identity()
        
        self.num_classes = num_classes
        self.shared_backbone = shared_backbone
        
        if shared_backbone:
            # Shared feature extraction layer (optional dimensionality reduction)
            # Each binary classifier then operates on the shared representation
            self.shared_features = nn.Identity()  # Can be replaced with nn.Linear if needed
            self.binary_classifiers = nn.ModuleList([
                nn.Linear(input_dim, 1) for _ in range(num_classes)
            ])
        else:
            # Independent binary classifiers (each processes input independently)
            self.binary_classifiers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, 1)
                )
                for _ in range(num_classes)
            ])
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through label-wise binary classifiers.
        
        Args:
            features: Input features of shape (B, D)
            
        Returns:
            Logits of shape (B, num_classes) - each column is a binary classifier output
        """
        x = self.preprocessing(features)
        
        if self.shared_backbone:
            shared = self.shared_features(x)
            logits = torch.cat([
                classifier(shared) for classifier in self.binary_classifiers
            ], dim=1)  # (B, num_classes)
        else:
            logits = torch.cat([
                classifier(x) for classifier in self.binary_classifiers
            ], dim=1)  # (B, num_classes)
        
        return logits

