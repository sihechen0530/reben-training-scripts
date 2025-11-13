"""
Late fusion strategies for combining features from multiple backbones.

This module implements various fusion strategies:
- concat: Simple concatenation
- weighted: Weighted sum with learnable weights
- linear_projection: Learnable linear projection followed by sum
"""
import torch
import torch.nn as nn
from typing import List, Optional
from abc import ABC, abstractmethod


class LateFusion(nn.Module, ABC):
    """
    Base class for late fusion strategies.
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
    
    @abstractmethod
    def get_output_dim(self, input_dims: List[int]) -> int:
        """
        Get the output dimension after fusion.
        
        Args:
            input_dims: List of input feature dimensions
            
        Returns:
            Output feature dimension
        """
        pass


class ConcatFusion(LateFusion):
    """
    Simple concatenation fusion strategy.
    
    Concatenates features from all backbones along the feature dimension.
    Output dimension = sum of all input dimensions.
    """
    
    def __init__(self, dim: int = 1):
        """
        Initialize concatenation fusion.
        
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
        Get the output dimension after concatenation.
        
        Args:
            input_dims: List of input feature dimensions
            
        Returns:
            Sum of all input dimensions
        """
        return sum(input_dims)


class WeightedSumFusion(LateFusion):
    """
    Weighted sum fusion strategy.
    
    Projects all features to the same dimension (max dimension) and
    performs a weighted sum with learnable or fixed weights.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        learnable_weights: bool = True,
        initial_weights: Optional[List[float]] = None,
    ):
        """
        Initialize weighted sum fusion.
        
        Args:
            feature_dims: List of feature dimensions for each backbone
            learnable_weights: Whether to use learnable weights (default: True)
            initial_weights: Initial weights for each backbone (default: uniform)
        """
        super().__init__()
        
        if len(feature_dims) == 0:
            raise ValueError("feature_dims must not be empty")
        
        self.num_backbones = len(feature_dims)
        self.max_dim = max(feature_dims)
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = [1.0 / self.num_backbones] * self.num_backbones
        
        if learnable_weights:
            self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
        else:
            self.register_buffer('weights', torch.tensor(initial_weights, dtype=torch.float32))
        
        # Project all features to the same dimension (max dimension)
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.max_dim) if dim != self.max_dim else nn.Identity()
            for dim in feature_dims
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Weighted sum of features.
        
        Args:
            features: List of feature tensors
            
        Returns:
            Weighted sum of features
        """
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} features, got {len(features)}"
            )
        
        # Project all features to same dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        
        # Normalize weights using softmax
        weights = torch.softmax(self.weights, dim=0)
        
        # Weighted sum
        fused = sum(w * feat for w, feat in zip(weights, projected))
        
        return fused
    
    def get_output_dim(self, input_dims: List[int]) -> int:
        """
        Get the output dimension after weighted sum.
        
        Args:
            input_dims: List of input feature dimensions
            
        Returns:
            Maximum of input dimensions
        """
        return max(input_dims) if input_dims else 0


class LinearProjectionFusion(LateFusion):
    """
    Learnable linear projection fusion strategy.
    
    Projects each feature to a common dimension and sums them.
    This is more flexible than weighted sum as it learns how to combine features.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: Optional[int] = None,
    ):
        """
        Initialize linear projection fusion.
        
        Args:
            feature_dims: List of feature dimensions for each backbone
            output_dim: Output dimension after fusion (default: max of input_dims)
        """
        super().__init__()
        
        if len(feature_dims) == 0:
            raise ValueError("feature_dims must not be empty")
        
        self.num_backbones = len(feature_dims)
        self.output_dim = output_dim if output_dim is not None else max(feature_dims)
        
        # Project all features to the output dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, self.output_dim)
            for dim in feature_dims
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Linear projection and sum of features.
        
        Args:
            features: List of feature tensors
            
        Returns:
            Sum of projected features
        """
        if len(features) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} features, got {len(features)}"
            )
        
        # Project all features to output dimension and sum
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        fused = sum(projected)
        
        return fused
    
    def get_output_dim(self, input_dims: List[int]) -> int:
        """
        Get the output dimension after linear projection.
        
        Args:
            input_dims: List of input feature dimensions (not used, returns self.output_dim)
            
        Returns:
            Output dimension
        """
        return self.output_dim

