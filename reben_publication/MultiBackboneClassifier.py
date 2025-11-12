"""
Multi-backbone model that concatenates features from multiple backbones with different input channels.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union
from configilm.ConfigILM import ILMConfiguration, ILMType
from configilm import ConfigILM

try:
    from reben_publication.DINOv3Backbone import DINOv3Backbone
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False


class BackboneConfig:
    """Configuration for a single backbone."""
    def __init__(
        self,
        name: str,
        input_channels: int,
        architecture: str = "resnet18",
        pretrained: bool = True,
        dinov3_model_name: Optional[str] = None,
    ):
        """
        Args:
            name: Unique name for this backbone
            input_channels: Number of input channels for this backbone
            architecture: Architecture name (timm model name or dinov3-*)
            pretrained: Whether to use pretrained weights
            dinov3_model_name: HuggingFace model name for DINOv3 (if architecture is dinov3-*)
        """
        self.name = name
        self.input_channels = input_channels
        self.architecture = architecture
        self.pretrained = pretrained
        self.dinov3_model_name = dinov3_model_name


class MultiBackboneFeatureExtractor(nn.Module):
    """
    Extracts features from multiple backbones with different input channels.
    Each backbone receives a subset of the input channels.
    """
    
    def __init__(
        self,
        backbone_configs: List[BackboneConfig],
        image_size: int = 120,
        num_classes: int = 19,
        drop_rate: float = 0.15,
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            backbone_configs: List of backbone configurations
            image_size: Input image size
            num_classes: Number of output classes
            drop_rate: Dropout rate
            drop_path_rate: Drop path rate
        """
        super().__init__()
        
        self.backbone_configs = backbone_configs
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Create backbones
        self.backbones = nn.ModuleDict()
        self.feature_dims = {}
        
        for config in backbone_configs:
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
                
                backbone = DINOv3Backbone(
                    model_name=dinov3_name,
                    num_classes=num_classes,
                    num_input_channels=config.input_channels,
                    image_size=image_size,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                    pretrained=config.pretrained,
                )
                # Get feature dimension from DINOv3
                feature_dim = backbone.embed_dim
                # Remove the classifier from DINOv3 backbone since we'll use our own
                backbone.classifier = nn.Identity()
                
            else:
                # Use ConfigILM for timm models
                config_ilm = ILMConfiguration(
                    network_type=ILMType.IMAGE_CLASSIFICATION,
                    classes=num_classes,
                    image_size=image_size,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                    timm_model_name=config.architecture,
                    channels=config.input_channels,
                )
                backbone = ConfigILM.ConfigILM(config_ilm)
                # Get feature dimension by running a dummy forward pass
                # ConfigILM wraps timm models, so we need to access the underlying model
                with torch.no_grad():
                    dummy_input = torch.randn(1, config.input_channels, image_size, image_size)
                    # Get features before classifier
                    # ConfigILM wraps timm model in self.model
                    if hasattr(backbone, 'model'):
                        timm_model = backbone.model
                        # For timm models, we can get features by removing the classifier
                        # Store original classifier
                        if hasattr(timm_model, 'classifier'):
                            original_classifier = timm_model.classifier
                            timm_model.classifier = nn.Identity()
                            # Get feature dimension
                            dummy_output = timm_model(dummy_input)
                            feature_dim = dummy_output.shape[-1] if len(dummy_output.shape) > 1 else dummy_output.shape[0]
                            # Restore classifier but we'll replace it later
                            timm_model.classifier = original_classifier
                        elif hasattr(timm_model, 'head'):
                            # Some timm models use 'head' instead of 'classifier'
                            original_head = timm_model.head
                            timm_model.head = nn.Identity()
                            dummy_output = timm_model(dummy_input)
                            feature_dim = dummy_output.shape[-1] if len(dummy_output.shape) > 1 else dummy_output.shape[0]
                            timm_model.head = original_head
                        else:
                            # Fallback: use full forward pass
                            dummy_output = backbone(dummy_input)
                            feature_dim = dummy_output.shape[-1]
                    else:
                        # Fallback
                        dummy_output = backbone(dummy_input)
                        feature_dim = dummy_output.shape[-1]
                
                # Replace classifier with identity to get features only
                if hasattr(backbone, 'model') and hasattr(backbone.model, 'classifier'):
                    backbone.model.classifier = nn.Identity()
                elif hasattr(backbone, 'model') and hasattr(backbone.model, 'head'):
                    backbone.model.head = nn.Identity()
                elif hasattr(backbone, 'classifier'):
                    backbone.classifier = nn.Identity()
            
            self.backbones[config.name] = backbone
            self.feature_dims[config.name] = feature_dim
            print(f"Created backbone '{config.name}': {config.architecture} with {config.input_channels} input channels, "
                  f"feature dim: {feature_dim}")
        
        # Total feature dimension
        self.total_feature_dim = sum(self.feature_dims.values())
        print(f"Total feature dimension after concatenation: {self.total_feature_dim}")
    
    def forward(self, x: torch.Tensor, channel_ranges: List[tuple]) -> torch.Tensor:
        """
        Forward pass through all backbones.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C is the total number of channels
            channel_ranges: List of (start, end) tuples indicating which channels each backbone should use
        
        Returns:
            Concatenated features of shape (B, total_feature_dim)
        """
        features = []
        
        for i, config in enumerate(self.backbone_configs):
            start_channel, end_channel = channel_ranges[i]
            # Extract channels for this backbone
            backbone_input = x[:, start_channel:end_channel, :, :]
            
            # Get features from backbone
            backbone = self.backbones[config.name]
            
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
                # Since we replaced classifier/head with Identity, this will return features
                feat = backbone(backbone_input)
            
            # Flatten if needed
            if len(feat.shape) > 2:
                feat = feat.view(feat.size(0), -1)
            
            features.append(feat)
        
        # Concatenate all features
        concatenated = torch.cat(features, dim=1)
        return concatenated


class MultiBackboneClassifier(nn.Module):
    """
    Multi-backbone classifier that concatenates features from multiple backbones
    and uses a linear classifier for classification.
    """
    
    def __init__(
        self,
        backbone_configs: List[BackboneConfig],
        image_size: int = 120,
        num_classes: int = 19,
        drop_rate: float = 0.15,
        drop_path_rate: float = 0.0,
        freeze_backbones: bool = False,
    ):
        """
        Args:
            backbone_configs: List of backbone configurations
            image_size: Input image size
            num_classes: Number of output classes
            drop_rate: Dropout rate for classifier
            drop_path_rate: Drop path rate (for backbones)
            freeze_backbones: If True, freeze all backbone parameters (only train classifier)
        """
        super().__init__()
        
        self.backbone_configs = backbone_configs
        self.image_size = image_size
        self.num_classes = num_classes
        self.freeze_backbones = freeze_backbones
        
        # Create feature extractor
        self.feature_extractor = MultiBackboneFeatureExtractor(
            backbone_configs=backbone_configs,
            image_size=image_size,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Freeze backbones if requested
        if freeze_backbones:
            for backbone in self.feature_extractor.backbones.values():
                for param in backbone.parameters():
                    param.requires_grad = False
            print("Frozen all backbone parameters")
        
        # Linear classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_extractor.total_feature_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.feature_extractor.total_feature_dim, num_classes)
        )
        
        # Compute channel ranges for each backbone
        self.channel_ranges = []
        current_channel = 0
        for config in backbone_configs:
            self.channel_ranges.append((current_channel, current_channel + config.input_channels))
            current_channel += config.input_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C is the sum of all backbone input channels
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Extract features from all backbones
        features = self.feature_extractor(x, self.channel_ranges)
        
        # Classify
        logits = self.classifier(features)
        
        return logits

