"""
Fusion strategies for combining features from multiple backbones.
"""
import torch
import torch.nn as nn
from typing import List, Optional
from abc import ABC, abstractmethod


class FusionStrategy(nn.Module, ABC):
    """
    Base class for fusion strategies.
    """
    
    @abstractmethod
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple backbones.
        
        Args:
            features: List of feature tensors, each of shape (B, D_i)
            
        Returns:
            Fused features of shape (B, D_fused)
        """
        pass


class ConcatenationFusion(FusionStrategy):
    """
    Concatenates features from multiple backbones (default fusion strategy).
    """
    
    def __init__(self, dim: int = 1):
        """
        Args:
            dim: Dimension along which to concatenate (default: 1 for feature dimension)
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate features along the specified dimension.
        
        Args:
            features: List of feature tensors, each of shape (B, D_i)
            
        Returns:
            Concatenated features of shape (B, sum(D_i))
        """
        if len(features) == 0:
            raise ValueError("No features to fuse")
        if len(features) == 1:
            return features[0]
        return torch.cat(features, dim=self.dim)
    
    def get_output_dim(self, input_dims: List[int]) -> int:
        """
        Get the output dimension after fusion.
        
        Args:
            input_dims: List of input feature dimensions
            
        Returns:
            Output feature dimension
        """
        return sum(input_dims)


class WeightedSumFusion(FusionStrategy):
    """
    Weighted sum fusion of features.
    """
    
    def __init__(self, feature_dims: List[int], learnable_weights: bool = True):
        """
        Args:
            feature_dims: List of feature dimensions for each backbone
            learnable_weights: Whether to use learnable weights (default: True)
        """
        super().__init__()
        self.num_backbones = len(feature_dims)
        
        if learnable_weights:
            self.weights = nn.Parameter(torch.ones(self.num_backbones) / self.num_backbones)
        else:
            self.register_buffer('weights', torch.ones(self.num_backbones) / self.num_backbones)
        
        # Project all features to the same dimension (use max dimension)
        max_dim = max(feature_dims)
        self.projections = nn.ModuleList([
            nn.Linear(dim, max_dim) if dim != max_dim else nn.Identity()
            for dim in feature_dims
        ])
        self.output_dim = max_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Weighted sum of features.
        
        Args:
            features: List of feature tensors
            
        Returns:
            Weighted sum of features
        """
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} features, got {len(features)}")
        
        # Project all features to same dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # Weighted sum
        weights = torch.softmax(self.weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, projected))
        
        return fused
    
    def get_output_dim(self, input_dims: List[int]) -> int:
        """
        Get the output dimension after fusion.
        
        Args:
            input_dims: List of input feature dimensions
            
        Returns:
            Output feature dimension (max of input dims)
        """
        return max(input_dims) if input_dims else 0

