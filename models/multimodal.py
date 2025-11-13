"""
Multimodal classifiers that fuse features from multiple backbones.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from models.base import BaseMultimodalClassifier
from models.fusion import ConcatenationFusion, FusionStrategy
from models.heads import LinearClassifierHead, ClassifierHead
from models.backbone_config import BackboneConfig
from models.backbones import create_backbone_from_config


class MultimodalLateFusionClassifier(BaseMultimodalClassifier):
    """
    Multimodal classifier with late fusion (default: concatenation + linear layer).
    """
    
    def __init__(
        self,
        backbone_configs: List[BackboneConfig],
        image_size: int = 120,
        num_classes: int = 19,
        drop_rate: float = 0.15,
        drop_path_rate: float = 0.0,
        freeze_backbones: bool = False,
        fusion: Optional[FusionStrategy] = None,
        classifier: Optional[ClassifierHead] = None,
    ):
        """
        Args:
            backbone_configs: List of backbone configurations
            image_size: Input image size
            num_classes: Number of output classes
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate
            freeze_backbones: If True, freeze all backbone parameters
            fusion: Fusion strategy (default: ConcatenationFusion)
            classifier: Classification head (default: LinearClassifierHead)
        """
        # Create backbones
        backbones = []
        feature_dims = []
        
        for config in backbone_configs:
            backbone, feature_dim = create_backbone_from_config(
                config=config,
                image_size=image_size,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
            backbones.append(backbone)
            feature_dims.append(feature_dim)
            print(f"Created backbone '{config.name}': {config.architecture} with {config.input_channels} input channels, "
                  f"feature dim: {feature_dim}")
        
        # Store freeze_backbones flag
        self.freeze_backbones = freeze_backbones
        
        # Freeze backbones if requested
        if freeze_backbones:
            for backbone in backbones:
                for param in backbone.parameters():
                    param.requires_grad = False
            print("Frozen all backbone parameters")
        
        # Create fusion (default: concatenation)
        if fusion is None:
            fusion = ConcatenationFusion()
            fused_dim = sum(feature_dims)
        else:
            if hasattr(fusion, 'get_output_dim'):
                fused_dim = fusion.get_output_dim(feature_dims)
            else:
                # Fallback: assume concatenation
                fused_dim = sum(feature_dims)
        
        # Create classifier (default: linear)
        if classifier is None:
            classifier = LinearClassifierHead(
                input_dim=fused_dim,
                num_classes=num_classes,
                drop_rate=drop_rate,
            )
        
        super().__init__(
            num_classes=num_classes,
            backbones=backbones,
            fusion=fusion,
            classifier=classifier,
        )
        
        self.backbone_configs = backbone_configs
        self.image_size = image_size
        self.feature_dims = feature_dims
        self.fused_dim = fused_dim
        
        # Compute channel ranges for each backbone
        self.channel_ranges = []
        current_channel = 0
        for config in backbone_configs:
            self.channel_ranges.append((current_channel, current_channel + config.input_channels))
            current_channel += config.input_channels
        
        print(f"Total feature dimension after fusion: {self.fused_dim}")
    
    def _extract_features(
        self,
        x: torch.Tensor,
        channel_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> List[torch.Tensor]:
        """
        Extract features from all backbones.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            channel_ranges: List of (start, end) tuples for channel assignment.
                          If None, uses self.channel_ranges.
        
        Returns:
            List of feature tensors from each backbone
        """
        if channel_ranges is None:
            channel_ranges = self.channel_ranges
        
        if len(channel_ranges) != len(self.backbones):
            raise ValueError(
                f"Mismatch: {len(channel_ranges)} channel ranges for {len(self.backbones)} backbones"
            )
        
        features = []
        
        for i, (backbone, config) in enumerate(zip(self.backbones, self.backbone_configs)):
            start_channel, end_channel = channel_ranges[i]
            # Extract channels for this backbone
            backbone_input = x[:, start_channel:end_channel, :, :]
            
            # Get features from backbone
            if hasattr(backbone, 'backbone'):
                # DINOv3 backbone
                try:
                    outputs = backbone.backbone(pixel_values=backbone_input)
                except TypeError:
                    outputs = backbone.backbone(backbone_input)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    feat = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    feat = outputs.last_hidden_state[:, 0]
                else:
                    feat = outputs
            else:
                # ConfigILM backbone (timm model)
                feat = backbone(backbone_input)
            
            # Flatten if needed
            if len(feat.shape) > 2:
                feat = feat.view(feat.size(0), -1)
            
            features.append(feat)
        
        return features

