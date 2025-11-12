"""
Classification heads for classifiers.
"""
import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod


class ClassifierHead(nn.Module, ABC):
    """
    Base class for classification heads.
    """
    
    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features of shape (B, D)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        pass


class LinearClassifierHead(ClassifierHead):
    """
    Linear classification head (default classifier).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 19,
        drop_rate: float = 0.15,
        use_layer_norm: bool = True,
    ):
        """
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
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 19,
        hidden_dims: Optional[list] = None,
        drop_rate: float = 0.15,
        use_layer_norm: bool = True,
        activation: str = "relu",
    ):
        """
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

