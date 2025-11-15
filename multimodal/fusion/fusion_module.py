"""
Late fusion strategies for combining features from multiple backbones.

This module implements various fusion strategies:
- concat: Simple concatenation
- weighted: Weighted sum with learnable weights
- linear_projection: Learnable linear projection followed by sum
- attention: Attention-based fusion with learnable queries
- bilinear: Bilinear pooling fusion for fine-grained interaction
- gated: Gated fusion with learnable gates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class AttentionFusion(LateFusion):
    """
    Attention-based fusion strategy.
    
    Uses learnable attention mechanism to dynamically weight and combine features
    from different backbones. Each backbone's features are projected to a common
    dimension and combined using attention scores.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: Optional[int] = None,
        num_heads: int = 1,
    ):
        """
        Initialize attention fusion.
        
        Args:
            feature_dims: List of feature dimensions for each backbone
            output_dim: Output dimension after fusion (default: max of input_dims)
            num_heads: Number of attention heads (default: 1 for multi-head attention)
        """
        super().__init__()
        
        if len(feature_dims) == 0:
            raise ValueError("feature_dims must not be empty")
        
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        
        self.num_backbones = len(feature_dims)
        self.num_heads = num_heads
        
        # Calculate output_dim, ensuring it's divisible by num_heads
        if output_dim is None:
            output_dim = max(feature_dims)
        # Round up to nearest multiple of num_heads
        if output_dim % num_heads != 0:
            output_dim = ((output_dim // num_heads) + 1) * num_heads
        
        self.output_dim = output_dim
        
        # Project all features to output dimension (as keys/values)
        self.key_projections = nn.ModuleList([
            nn.Linear(dim, self.output_dim)
            for dim in feature_dims
        ])
        self.value_projections = nn.ModuleList([
            nn.Linear(dim, self.output_dim)
            for dim in feature_dims
        ])
        
        # Learnable query for attention
        self.query = nn.Parameter(torch.randn(1, self.num_heads, self.output_dim))
        
        # Output projection
        self.output_proj = nn.Linear(self.output_dim, self.output_dim)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Attention-based fusion of features.
        
        Args:
            features: List of feature tensors, each of shape (B, D_i)
            
        Returns:
            Fused features of shape (B, output_dim)
        """
        if len(features) != len(self.key_projections):
            raise ValueError(
                f"Expected {len(self.key_projections)} features, got {len(features)}"
            )
        
        batch_size = features[0].shape[0]
        
        # Project features to keys and values
        keys = [proj(feat) for proj, feat in zip(self.key_projections, features)]
        values = [proj(feat) for proj, feat in zip(self.value_projections, features)]
        
        # Stack keys and values: (B, num_backbones, output_dim)
        keys = torch.stack(keys, dim=1)  # (B, num_backbones, output_dim)
        values = torch.stack(values, dim=1)  # (B, num_backbones, output_dim)
        
        # Expand query: (1, num_heads, output_dim) -> (B, num_heads, output_dim)
        query = self.query.expand(batch_size, -1, -1)
        
        # Reshape for multi-head attention: (B, num_heads, 1, output_dim // num_heads)
        head_dim = self.output_dim // self.num_heads
        query = query.reshape(batch_size, self.num_heads, 1, head_dim)
        keys = keys.reshape(batch_size, self.num_backbones, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, num_backbones, head_dim)
        values = values.reshape(batch_size, self.num_backbones, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, num_backbones, head_dim)
        
        # Compute attention scores: (B, num_heads, 1, num_backbones)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / (head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values: (B, num_heads, 1, head_dim)
        attended = torch.matmul(attention_weights, values)
        
        # Reshape and project: (B, output_dim)
        attended = attended.reshape(batch_size, self.output_dim)
        fused = self.output_proj(attended)
        
        return fused
    
    def get_output_dim(self, input_dims: List[int]) -> int:
        """
        Get the output dimension after attention fusion.
        
        Args:
            input_dims: List of input feature dimensions (not used, returns self.output_dim)
            
        Returns:
            Output dimension
        """
        return self.output_dim


class BilinearFusion(LateFusion):
    """
    Bilinear pooling fusion strategy.
    
    Computes bilinear interactions between features from different backbones.
    This is useful for fine-grained feature interactions, commonly used in
    multimodal learning and fine-grained classification tasks.
    
    Note: Output dimension = output_dim^2, which can be large. Consider using
    compact bilinear pooling or low-rank approximation for efficiency.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: Optional[int] = None,
        use_low_rank: bool = True,
        low_rank_dim: Optional[int] = None,
    ):
        """
        Initialize bilinear fusion.
        
        Args:
            feature_dims: List of feature dimensions for each backbone
            output_dim: Output dimension for each feature after projection (default: max of input_dims)
            use_low_rank: Whether to use low-rank approximation for efficiency (default: True)
            low_rank_dim: Low-rank dimension for approximation (default: output_dim if use_low_rank)
        """
        super().__init__()
        
        if len(feature_dims) < 2:
            raise ValueError("Bilinear fusion requires at least 2 features")
        
        if len(feature_dims) > 2:
            raise ValueError(
                "Bilinear fusion currently supports only 2 features. "
                "For more features, consider using other fusion methods."
            )
        
        self.output_dim = output_dim if output_dim is not None else max(feature_dims)
        self.use_low_rank = use_low_rank
        
        if use_low_rank:
            self.low_rank_dim = low_rank_dim if low_rank_dim is not None else self.output_dim
            # Low-rank projection matrices
            self.proj1 = nn.Linear(feature_dims[0], self.low_rank_dim)
            self.proj2 = nn.Linear(feature_dims[1], self.low_rank_dim)
            # Output projection
            self.output_proj = nn.Linear(self.low_rank_dim, self.output_dim)
            self._output_dim = self.output_dim
        else:
            # Full bilinear: project both to output_dim, then compute outer product
            self.proj1 = nn.Linear(feature_dims[0], self.output_dim)
            self.proj2 = nn.Linear(feature_dims[1], self.output_dim)
            # Output projection from flattened outer product
            self.output_proj = nn.Linear(self.output_dim * self.output_dim, self.output_dim)
            self._output_dim = self.output_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Bilinear fusion of features.
        
        Args:
            features: List of 2 feature tensors, each of shape (B, D_i)
            
        Returns:
            Fused features of shape (B, output_dim)
        """
        if len(features) != 2:
            raise ValueError(f"Bilinear fusion requires exactly 2 features, got {len(features)}")
        
        feat1, feat2 = features[0], features[1]
        
        # Project features
        proj1 = self.proj1(feat1)  # (B, output_dim or low_rank_dim)
        proj2 = self.proj2(feat2)  # (B, output_dim or low_rank_dim)
        
        if self.use_low_rank:
            # Low-rank bilinear: element-wise product
            bilinear = proj1 * proj2  # (B, low_rank_dim)
            fused = self.output_proj(bilinear)  # (B, output_dim)
        else:
            # Full bilinear: outer product
            # (B, output_dim, 1) * (B, 1, output_dim) -> (B, output_dim, output_dim)
            bilinear = torch.bmm(proj1.unsqueeze(2), proj2.unsqueeze(1))
            # Flatten and project
            bilinear_flat = bilinear.reshape(bilinear.shape[0], -1)  # (B, output_dim^2)
            fused = self.output_proj(bilinear_flat)  # (B, output_dim)
        
        return fused
    
    def get_output_dim(self, input_dims: List[int]) -> int:
        """
        Get the output dimension after bilinear fusion.
        
        Args:
            input_dims: List of input feature dimensions (not used, returns self._output_dim)
            
        Returns:
            Output dimension
        """
        return self._output_dim


class GatedFusion(LateFusion):
    """
    Gated fusion strategy.
    
    Uses learnable gates to control the contribution of each backbone's features.
    The gates are computed based on the features themselves, allowing adaptive
    fusion based on input content.
    """
    
    def __init__(
        self,
        feature_dims: List[int],
        output_dim: Optional[int] = None,
    ):
        """
        Initialize gated fusion.
        
        Args:
            feature_dims: List of feature dimensions for each backbone
            output_dim: Output dimension after fusion (default: max of input_dims)
        """
        super().__init__()
        
        if len(feature_dims) == 0:
            raise ValueError("feature_dims must not be empty")
        
        self.num_backbones = len(feature_dims)
        self.output_dim = output_dim if output_dim is not None else max(feature_dims)
        
        # Project all features to output dimension
        self.feature_projections = nn.ModuleList([
            nn.Linear(dim, self.output_dim)
            for dim in feature_dims
        ])
        
        # Gate networks: each computes a gate score for its corresponding feature
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.output_dim),
                nn.ReLU(),
                nn.Linear(self.output_dim, self.output_dim),
                nn.Sigmoid(),  # Gate values between 0 and 1
            )
            for dim in feature_dims
        ])
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Gated fusion of features.
        
        Args:
            features: List of feature tensors, each of shape (B, D_i)
            
        Returns:
            Fused features of shape (B, output_dim)
        """
        if len(features) != len(self.feature_projections):
            raise ValueError(
                f"Expected {len(self.feature_projections)} features, got {len(features)}"
            )
        
        # Project features and compute gates
        projected_features = []
        gates = []
        
        for feat, feat_proj, gate_net in zip(features, self.feature_projections, self.gate_networks):
            projected = feat_proj(feat)  # (B, output_dim)
            gate = gate_net(feat)  # (B, output_dim)
            
            projected_features.append(projected)
            gates.append(gate)
        
        # Normalize gates so they sum to 1 across backbones
        gates_stack = torch.stack(gates, dim=0)  # (num_backbones, B, output_dim)
        gates_normalized = F.softmax(gates_stack, dim=0)  # Normalize across backbones
        gates_normalized = gates_normalized.transpose(0, 1)  # (B, num_backbones, output_dim)
        
        # Apply gates and sum
        projected_stack = torch.stack(projected_features, dim=1)  # (B, num_backbones, output_dim)
        gated_features = projected_stack * gates_normalized  # (B, num_backbones, output_dim)
        fused = gated_features.sum(dim=1)  # (B, output_dim)
        
        return fused
    
    def get_output_dim(self, input_dims: List[int]) -> int:
        """
        Get the output dimension after gated fusion.
        
        Args:
            input_dims: List of input feature dimensions (not used, returns self.output_dim)
            
        Returns:
            Output dimension
        """
        return self.output_dim

