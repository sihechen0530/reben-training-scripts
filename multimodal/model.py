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
    2. S2 non-RGB (11 channels) + S1 (2 channels) -> ResNet101 backbone
    3. Late fusion of embeddings
    4. Classification head
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
        
        if resnet_backbone is None:
            resnet_config = config["backbones"]["resnet101"]
            self.resnet_backbone = ResNetBackbone(
                input_channels=13,  # 11 S2 non-RGB + 2 S1
                pretrained=resnet_config.get("pretrained", True),
                freeze=resnet_config.get("freeze", False),
                image_size=self.image_size,
            )
        else:
            self.resnet_backbone = resnet_backbone
        
        # Get feature dimensions
        dinov3_dim = self.dinov3_backbone.get_embed_dim()
        resnet_dim = self.resnet_backbone.get_embed_dim()
        feature_dims = [dinov3_dim, resnet_dim]
        
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
        self.resnet_lr = config["backbones"]["resnet101"].get("lr", 1e-4)
        
        print(f"MultiModalModel initialized:")
        print(f"  - DINOv3 backbone: {dinov3_dim} dim (frozen={self.dinov3_backbone.freeze})")
        print(f"  - ResNet101 backbone: {resnet_dim} dim (frozen={self.resnet_backbone.freeze})")
        print(f"  - Fusion: {config['fusion']['type']} -> {fused_dim} dim")
        print(f"  - Classifier: {config['classifier']['type']} -> {num_classes} classes")
    
    def forward(
        self,
        s1_data: torch.Tensor,
        s2_data: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through the multimodal model.
        
        Args:
            s1_data: S1 radar data of shape (B, 2, H, W)
            s2_data: S2 spectral data of shape (B, 14, H, W)
                     Channels: [RGB (3), non-RGB (11)]
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
        # Extract S2 RGB (first 3 channels)
        s2_rgb = s2_data[:, :3, :, :]  # (B, 3, H, W)
        
        # Extract S2 non-RGB (channels 3-13) and concatenate with S1
        s2_non_rgb = s2_data[:, 3:, :, :]  # (B, 11, H, W)
        s1_s2_combined = torch.cat([s2_non_rgb, s1_data], dim=1)  # (B, 13, H, W)
        
        # Extract features from both backbones
        dinov3_features = self.dinov3_backbone(s2_rgb)  # (B, dinov3_dim)
        resnet_features = self.resnet_backbone(s1_s2_combined)  # (B, resnet_dim)
        
        # Fuse features
        fused_features = self.fusion([dinov3_features, resnet_features])  # (B, fused_dim)
        
        # Classification
        logits = self.classifier(fused_features)  # (B, num_classes)
        
        if return_embeddings:
            embeddings = {
                "dinov3": dinov3_features,
                "resnet": resnet_features,
                "fused": fused_features,
            }
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
        if not self.resnet_backbone.freeze:
            groups.append({
                "params": self.resnet_backbone.parameters(),
                "lr": self.resnet_lr,
                "name": "resnet101_backbone",
            })
        
        # Fusion parameters (use average of backbone LRs or default)
        fusion_lr = (self.dinov3_lr + self.resnet_lr) / 2
        groups.append({
            "params": self.fusion.parameters(),
            "lr": fusion_lr,
            "name": "fusion",
        })
        
        # Classifier parameters (use average of backbone LRs or default)
        classifier_lr = (self.dinov3_lr + self.resnet_lr) / 2
        groups.append({
            "params": self.classifier.parameters(),
            "lr": classifier_lr,
            "name": "classifier",
        })
        
        return groups

