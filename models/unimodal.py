"""
Unimodal classifiers that use a single backbone.
"""
import torch
import torch.nn as nn
from typing import Optional

from models.base import BaseUnimodalClassifier
from models.heads import LinearClassifierHead
from models.backbones import create_resnet_backbone, create_dinov3_backbone

try:
    from reben_publication.DINOv3Backbone import DINOv3Backbone
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False


class ResNetClassifier(BaseUnimodalClassifier):
    """
    ResNet classifier using ConfigILM/timm models.
    """
    
    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 19,
        num_channels: int = 3,
        image_size: int = 120,
        drop_rate: float = 0.15,
        drop_path_rate: float = 0.0,
        pretrained: bool = True,
    ):
        """
        Args:
            architecture: ResNet architecture name (timm model name)
            num_classes: Number of output classes
            num_channels: Number of input channels
            image_size: Input image size
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate
            pretrained: Whether to use pretrained weights
        """
        # Create backbone using factory
        backbone, feature_dim = create_resnet_backbone(
            architecture=architecture,
            input_channels=num_channels,
            image_size=image_size,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
        )
        
        # Create linear classifier head
        classifier = LinearClassifierHead(
            input_dim=feature_dim,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        
        super().__init__(num_classes=num_classes, backbone=backbone, classifier=classifier)
        self.feature_dim = feature_dim
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ResNet backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Features of shape (B, feature_dim)
        """
        features = self.backbone(x)
        # Flatten if needed
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        return features


class DINOv3LinearProbeClassifier(BaseUnimodalClassifier):
    """
    DINOv3 backbone with linear probe classifier.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        num_classes: int = 19,
        num_input_channels: int = 3,
        image_size: int = 120,
        drop_rate: float = 0.15,
        drop_path_rate: float = 0.0,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace DINOv3 model name
            num_classes: Number of output classes
            num_input_channels: Number of input channels
            image_size: Input image size
            drop_rate: Dropout rate for classifier
            drop_path_rate: Drop path rate
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone (linear probing mode)
        """
        if not DINOV3_AVAILABLE:
            raise ImportError(
                "DINOv3 not available. Install transformers: pip install transformers"
            )
        
        # Create DINOv3 backbone using factory
        backbone, feature_dim = create_dinov3_backbone(
            model_name=model_name,
            input_channels=num_input_channels,
            image_size=image_size,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained,
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in backbone.backbone.parameters():
                param.requires_grad = False
            # Keep input adaptation layer trainable if present
            try:
                if hasattr(backbone.backbone, 'embeddings') and hasattr(backbone.backbone.embeddings, 'patch_embeddings'):
                    pe = backbone.backbone.embeddings.patch_embeddings
                    if hasattr(pe, 'projection'):
                        proj = pe.projection
                    elif isinstance(pe, torch.nn.Conv2d):
                        proj = pe
                    else:
                        proj = None
                    if proj is not None:
                        for param in proj.parameters():
                            param.requires_grad = True
            except Exception:
                pass
        
        # Create linear classifier head
        classifier = LinearClassifierHead(
            input_dim=feature_dim,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        
        super().__init__(num_classes=num_classes, backbone=backbone, classifier=classifier)
        self.feature_dim = feature_dim
        self.freeze_backbone = freeze_backbone
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from DINOv3 backbone.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Features of shape (B, embed_dim)
        """
        # Get features from DINOv3 backbone
        try:
            outputs = self.backbone.backbone(pixel_values=x)
        except TypeError:
            outputs = self.backbone.backbone(x)
        
        # Extract features
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, torch.Tensor):
            features = outputs[:, 0] if len(outputs.shape) > 2 else outputs
        else:
            raise ValueError(f"Unexpected output type from DINOv3 backbone: {type(outputs)}")
        
        return features

