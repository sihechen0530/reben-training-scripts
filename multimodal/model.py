"""
Main multimodal model for combining S1 and S2 satellite data.

This module provides the MultiModalModel class that:
- Processes S2 RGB data (3 channels) through DINOv3 backbone
- Processes S2 non-RGB (11 channels) + S1 (2 channels) through ResNet101 backbone
- Fuses features using configurable late fusion strategies
- Applies classification head to produce final predictions
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any, Union

from multimodal.backbones.dinov3_backbone import DINOv3Backbone
from multimodal.backbones.resnet_backbone import ResNetBackbone
from multimodal.fusion.fusion_module import (
    LateFusion,
    ConcatFusion,
    WeightedSumFusion,
    LinearProjectionFusion,
)
from multimodal.classifier.classification_head import (
    ClassifierHead,
    LinearClassifierHead,
    MLPClassifierHead,
)


class MultiModalModel(nn.Module):
    """
    Multimodal classification model for S1 and S2 satellite data.
    
    Architecture:
    1. S2 RGB (3 channels) -> DINOv3 backbone
    2. S2 non-RGB (9 channels) + optionally S1 (2 channels) -> ResNet101 backbone
    3. Late fusion of embeddings
    4. Classification head
    
    The ResNet input channels are configurable:
    - If use_s1=True: 9 S2 non-RGB + 2 S1 = 11 channels
    - If use_s1=False: 9 S2 non-RGB only = 9 channels
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dinov3_backbone: Optional[DINOv3Backbone] = None,
        resnet_backbone: Optional[ResNetBackbone] = None,
        fusion: Optional[LateFusion] = None,
        classifier: Optional[ClassifierHead] = None,
    ):
        """
        Initialize multimodal model.
        
        Args:
            config: Configuration dictionary with the following structure:
                {
                    "backbones": {
                        "dinov3": {
                            "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
                            "pretrained": True,
                            "freeze": False,
                            "lr": 1e-4,
                        },
                        "resnet101": {
                            "input_channels": 11,  # 9 S2 non-RGB + 2 S1 (or 9 if use_s1=False)
                            "pretrained": True,
                            "freeze": False,
                            "lr": 1e-4,
                        },
                    },
                    "fusion": {
                        "type": "concat",  # "concat", "weighted", or "linear_projection"
                        "output_dim": None,  # Only used for linear_projection
                    },
                    "classifier": {
                        "type": "linear",  # "linear" or "mlp"
                        "hidden_dim": 512,  # Only used for mlp
                        "num_classes": 19,
                        "drop_rate": 0.15,
                    },
                    "image_size": 120,
                }
            dinov3_backbone: Optional pre-initialized DINOv3 backbone
            resnet_backbone: Optional pre-initialized ResNet101 backbone
            fusion: Optional pre-initialized fusion module
            classifier: Optional pre-initialized classifier head
        """
        super().__init__()
        
        # Default configuration
        default_config = {
            "backbones": {
                "dinov3": {
                    "model_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
                    "pretrained": True,
                    "freeze": False,
                    "lr": 1e-4,
                },
                "resnet101": {
                    "pretrained": True,
                    "freeze": False,
                    "lr": 1e-4,
                    "enabled": True,
                },
            },
            "fusion": {
                "type": "concat",
            },
            "classifier": {
                "type": "linear",
                "num_classes": 19,
                "drop_rate": 0.15,
            },
            "image_size": 120,
        }
        
        if config is None:
            config = default_config
        else:
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config[key], dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
        
        self.config = config
        self.image_size = config.get("image_size", 120)
        resnet_config = config.get("backbones", {}).get("resnet101", {})
        self.use_resnet = resnet_config.get("enabled", True)
        
        # Note: RGB channel separation is now done in the Lightning module.
        # The model receives pre-separated rgb_data and non_rgb_s1_data,
        # so we don't need to configure RGB channel indices here anymore.
        # However, we keep the config for reference/documentation.
        self.rgb_band_names = config.get("rgb_band_names", None)
        if self.rgb_band_names:
            print(f"RGB band names (for reference): {self.rgb_band_names}")
        
        # Initialize backbones
        if dinov3_backbone is None:
            dinov3_config = config["backbones"]["dinov3"]
            self.dinov3_backbone = DINOv3Backbone(
                model_name=dinov3_config.get("model_name", "facebook/dinov3-vits16-pretrain-lvd1689m"),
                pretrained=dinov3_config.get("pretrained", True),
                freeze=dinov3_config.get("freeze", False),
                input_channels=3,  # RGB only
                image_size=self.image_size,
            )
        else:
            self.dinov3_backbone = dinov3_backbone
        
        self.resnet_backbone = None
        if self.use_resnet:
            if resnet_backbone is None:
                resnet_settings = config.get("backbones", {}).get("resnet101")
                if resnet_settings is None:
                    raise ValueError("ResNet branch is enabled but no 'resnet101' config was provided.")
                # Get input_channels from config (defaults to 11 for backward compatibility)
                # 9 S2 non-RGB + 2 S1 if use_s1=True, or 9 S2 non-RGB only if use_s1=False
                input_channels = resnet_settings.get("input_channels", 11)
                self.resnet_backbone = ResNetBackbone(
                    input_channels=input_channels,
                    pretrained=resnet_settings.get("pretrained", True),
                    freeze=resnet_settings.get("freeze", False),
                    image_size=self.image_size,
                )
            else:
                self.resnet_backbone = resnet_backbone
        
        # Get feature dimensions
        dinov3_dim = self.dinov3_backbone.get_embed_dim()
        feature_dims = [dinov3_dim]
        if self.use_resnet and self.resnet_backbone is not None:
            resnet_dim = self.resnet_backbone.get_embed_dim()
            feature_dims.append(resnet_dim)
        else:
            resnet_dim = None
        
        # Initialize fusion
        if fusion is None:
            fusion_config = config["fusion"]
            fusion_type = fusion_config.get("type", "concat")
            
            if fusion_type == "concat":
                self.fusion = ConcatFusion()
                fused_dim = self.fusion.get_output_dim(feature_dims)
            elif fusion_type == "weighted":
                self.fusion = WeightedSumFusion(
                    feature_dims=feature_dims,
                    learnable_weights=True,
                )
                fused_dim = self.fusion.get_output_dim(feature_dims)
            elif fusion_type == "linear_projection":
                output_dim = fusion_config.get("output_dim", max(feature_dims))
                self.fusion = LinearProjectionFusion(
                    feature_dims=feature_dims,
                    output_dim=output_dim,
                )
                fused_dim = self.fusion.get_output_dim(feature_dims)
            else:
                raise ValueError(
                    f"Unknown fusion type: {fusion_type}. "
                    "Must be 'concat', 'weighted', or 'linear_projection'"
                )
        else:
            self.fusion = fusion
            fused_dim = self.fusion.get_output_dim(feature_dims)
        
        # Initialize classifier
        if classifier is None:
            classifier_config = config["classifier"]
            classifier_type = classifier_config.get("type", "linear")
            num_classes = classifier_config.get("num_classes", 19)
            drop_rate = classifier_config.get("drop_rate", 0.15)
            
            if classifier_type == "linear":
                self.classifier = LinearClassifierHead(
                    input_dim=fused_dim,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                )
            elif classifier_type == "mlp":
                hidden_dim = classifier_config.get("hidden_dim", 512)
                self.classifier = MLPClassifierHead(
                    input_dim=fused_dim,
                    num_classes=num_classes,
                    hidden_dims=[hidden_dim],
                    drop_rate=drop_rate,
                )
            else:
                raise ValueError(
                    f"Unknown classifier type: {classifier_type}. "
                    "Must be 'linear' or 'mlp'"
                )
        else:
            self.classifier = classifier
        
        # Store for parameter grouping (different learning rates)
        self.dinov3_lr = config["backbones"]["dinov3"].get("lr", 1e-4)
        self.resnet_lr = resnet_config.get("lr", 1e-4)
        
        print(f"MultiModalModel initialized:")
        print(f"  - DINOv3 backbone: {dinov3_dim} dim (frozen={self.dinov3_backbone.freeze})")
        if self.use_resnet and self.resnet_backbone is not None:
            print(f"  - ResNet101 backbone: {resnet_dim} dim (frozen={self.resnet_backbone.freeze})")
        else:
            print("  - ResNet101 backbone: DISABLED")
        print(f"  - Fusion: {config['fusion']['type']} -> {fused_dim} dim")
        print(f"  - Classifier: {config['classifier']['type']} -> {num_classes} classes")
    
    def forward(
        self,
        rgb_data: torch.Tensor,
        non_rgb_s1_data: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the multimodal model.
        
        Args:
            rgb_data: RGB channels (S2 RGB) of shape (B, 3, H, W)
            non_rgb_s1_data: Combined S2 non-RGB + optionally S1 data of shape (B, C, H, W)
                           where C = 9 (S2_non_RGB only) or 11 (S2_non_RGB + S1).
                           Can be None when the ResNet branch is disabled.
            return_embeddings: If True, also return individual embeddings
        
        Returns:
            If return_embeddings=False:
                logits: Classification logits of shape (B, num_classes)
            If return_embeddings=True:
                (logits, embeddings_dict) where embeddings_dict contains:
                    - "dinov3": DINOv3 embeddings
                    - "resnet": ResNet embeddings
                    - "fused": Fused embeddings
        """
        
        # Extract features from both backbones
        dinov3_features = self.dinov3_backbone(rgb_data)  # (B, dinov3_dim)
        features = [dinov3_features]
        embeddings: Dict[str, torch.Tensor] = {
            "dinov3": dinov3_features,
        }

        if self.use_resnet:
            if non_rgb_s1_data is None:
                raise ValueError("ResNet branch is enabled but non_rgb_s1_data was not provided.")
            if non_rgb_s1_data.numel() == 0:
                raise ValueError("ResNet branch is enabled but received empty non_rgb_s1_data tensor.")
            resnet_features = self.resnet_backbone(non_rgb_s1_data)  # (B, resnet_dim)
            features.append(resnet_features)
            embeddings["resnet"] = resnet_features
        
        # Fuse features
        fused_features = self.fusion(features)  # (B, fused_dim)
        
        # Classification
        logits = self.classifier(fused_features)  # (B, num_classes)
        
        if return_embeddings:
            embeddings["fused"] = fused_features
            return logits, embeddings
        else:
            return logits
    
    def get_parameter_groups(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups for different learning rates.
        
        This is useful for training with different learning rates for each backbone.
        
        Returns:
            List of parameter group dictionaries for optimizer
        """
        groups = []
        
        # DINOv3 backbone parameters
        if not self.dinov3_backbone.freeze:
            groups.append({
                "params": self.dinov3_backbone.parameters(),
                "lr": self.dinov3_lr,
                "name": "dinov3_backbone",
            })
        
        # ResNet101 backbone parameters
        if self.use_resnet and self.resnet_backbone is not None and not self.resnet_backbone.freeze:
            groups.append({
                "params": self.resnet_backbone.parameters(),
                "lr": self.resnet_lr,
                "name": "resnet101_backbone",
            })
        
        # Fusion parameters (use average of backbone LRs or default)
        lr_values = [self.dinov3_lr]
        if self.use_resnet:
            lr_values.append(self.resnet_lr)
        fusion_lr = sum(lr_values) / len(lr_values)
        groups.append({
            "params": self.fusion.parameters(),
            "lr": fusion_lr,
            "name": "fusion",
        })
        
        # Classifier parameters (use average of backbone LRs or default)
        classifier_lr = fusion_lr
        groups.append({
            "params": self.classifier.parameters(),
            "lr": classifier_lr,
            "name": "classifier",
        })
        
        return groups

