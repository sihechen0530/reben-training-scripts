"""
Factory functions for creating backbone models.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from configilm.ConfigILM import ILMConfiguration, ILMType
from configilm import ConfigILM

from models.backbone_config import BackboneConfig

try:
    from reben_publication.DINOv3Backbone import DINOv3Backbone
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False
    DINOv3Backbone = None


def create_resnet_backbone(
    architecture: str,
    input_channels: int,
    image_size: int,
    num_classes: int,
    drop_rate: float = 0.15,
    drop_path_rate: float = 0.0,
    pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Create a ResNet backbone using ConfigILM/timm models.
    
    Args:
        architecture: ResNet architecture name (timm model name)
        input_channels: Number of input channels
        image_size: Input image size
        num_classes: Number of classes (for feature dimension calculation)
        drop_rate: Dropout rate
        drop_path_rate: Drop path rate
        pretrained: Whether to use pretrained weights
        
    Returns:
        Tuple of (backbone, feature_dim)
    """
    config_ilm = ILMConfiguration(
        network_type=ILMType.IMAGE_CLASSIFICATION,
        classes=num_classes,
        image_size=image_size,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        timm_model_name=architecture,
        channels=input_channels,
    )
    backbone = ConfigILM.ConfigILM(config_ilm)
    
    # Get feature dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, input_channels, image_size, image_size)
        dummy_output = backbone(dummy_input)
        feature_dim = dummy_output.shape[-1]
    
    # Replace classifier with identity to get features only
    if hasattr(backbone, 'model') and hasattr(backbone.model, 'classifier'):
        backbone.model.classifier = nn.Identity()
    elif hasattr(backbone, 'model') and hasattr(backbone.model, 'head'):
        backbone.model.head = nn.Identity()
    elif hasattr(backbone, 'classifier'):
        backbone.classifier = nn.Identity()
    
    return backbone, feature_dim


def create_dinov3_backbone(
    model_name: str,
    input_channels: int,
    image_size: int,
    num_classes: int,
    drop_rate: float = 0.15,
    drop_path_rate: float = 0.0,
    pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Create a DINOv3 backbone.
    
    Args:
        model_name: HuggingFace DINOv3 model name
        input_channels: Number of input channels
        image_size: Input image size
        num_classes: Number of classes (for feature dimension calculation)
        drop_rate: Dropout rate
        drop_path_rate: Drop path rate
        pretrained: Whether to use pretrained weights
        
    Returns:
        Tuple of (backbone, feature_dim)
    """
    if not DINOV3_AVAILABLE:
        raise ImportError(
            "DINOv3 not available. Install transformers: pip install transformers"
        )
    
    backbone = DINOv3Backbone(
        model_name=model_name,
        num_classes=num_classes,
        num_input_channels=input_channels,
        image_size=image_size,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        pretrained=pretrained,
    )
    feature_dim = backbone.embed_dim
    # Remove classifier from backbone
    backbone.classifier = nn.Identity()
    
    return backbone, feature_dim


def create_backbone_from_config(
    config: BackboneConfig,
    image_size: int,
    num_classes: int,
    drop_rate: float = 0.15,
    drop_path_rate: float = 0.0,
) -> Tuple[nn.Module, int]:
    """
    Create a backbone from configuration.
    
    Args:
        config: Backbone configuration
        image_size: Input image size
        num_classes: Number of classes (for feature dimension calculation)
        drop_rate: Dropout rate
        drop_path_rate: Drop path rate
        
    Returns:
        Tuple of (backbone, feature_dim)
    """
    # Determine if using DINOv3
    use_dinov3 = (
        DINOV3_AVAILABLE and 
        (config.dinov3_model_name is not None or
         config.architecture.startswith('dinov3'))
    )
    
    if use_dinov3:
        # Determine DINOv3 model name
        if config.dinov3_model_name is None:
            if 'small' in config.architecture.lower() or 's' in config.architecture.lower():
                dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
            elif 'base' in config.architecture.lower() or 'b' in config.architecture.lower():
                dinov3_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            elif 'large' in config.architecture.lower() or 'l' in config.architecture.lower():
                dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            elif 'giant' in config.architecture.lower() or 'g' in config.architecture.lower():
                dinov3_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
            else:
                dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        else:
            dinov3_name = config.dinov3_model_name
        
        return create_dinov3_backbone(
            model_name=dinov3_name,
            input_channels=config.input_channels,
            image_size=image_size,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=config.pretrained,
        )
    else:
        # Use ResNet/timm models
        return create_resnet_backbone(
            architecture=config.architecture,
            input_channels=config.input_channels,
            image_size=image_size,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=config.pretrained,
        )

