"""
ResNet backbone implementation for multi-channel spectral/radar data.

This module provides a ResNet101 backbone that can handle custom input channels
(e.g., 11 S2 non-RGB channels + 2 S1 channels = 13 channels total).
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class ResNetBackbone(nn.Module):
    """
    ResNet101 backbone for processing multi-channel spectral/radar data.
    
    This backbone can handle custom input channels (e.g., 13 channels for
    S2 non-RGB + S1 data) and can be frozen or fine-tuned during training.
    """
    
    def __init__(
        self,
        input_channels: int = 9,
        pretrained: bool = True,
        freeze: bool = False,
        image_size: int = 120,
    ):
        """
        Initialize ResNet101 backbone.
        
        Args:
            input_channels: Number of input channels (e.g., 13 for S2 non-RGB + S1)
            pretrained: Whether to load pretrained ImageNet weights
            freeze: Whether to freeze backbone parameters
            image_size: Input image size (for compatibility)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.image_size = image_size
        self.freeze = freeze
        
        # Load pretrained ResNet101
        print(f"Loading ResNet101 (pretrained={pretrained}, input_channels={input_channels})")
        if pretrained:
            resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet101(weights=None)
        
        # Replace first conv layer to handle custom input channels
        if input_channels != 3:
            # Create new conv layer
            original_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize new conv layer
            # Copy weights from first channel and average for remaining channels
            if pretrained and input_channels >= 3:
                with torch.no_grad():
                    # Average the 3 RGB channels and replicate for all input channels
                    avg_weight = original_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.data = avg_weight.repeat(1, input_channels, 1, 1)
                    if new_conv.bias is not None:
                        new_conv.bias.data = original_conv.bias.data.clone()
            else:
                # Use default initialization
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
                if new_conv.bias is not None:
                    nn.init.constant_(new_conv.bias, 0)
            
            resnet.conv1 = new_conv
            print(f"Adapted first conv layer for {input_channels} input channels")
        
        # Remove the final fully connected layer (we only want features)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, image_size, image_size)
            dummy_output = self.backbone(dummy_input)
            self.feature_dim = dummy_output.view(1, -1).shape[1]
        
        print(f"ResNet101 feature dimension: {self.feature_dim}")
        
        # Freeze parameters if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("ResNet101 backbone parameters frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet101 backbone.
        
        Args:
            x: Input tensor of shape (B, input_channels, H, W)
            
        Returns:
            Feature embeddings of shape (B, feature_dim)
        """
        features = self.backbone(x)
        # Flatten spatial dimensions
        features = features.view(features.size(0), -1)
        return features
    
    def get_embed_dim(self) -> int:
        """
        Get the embedding dimension of this backbone.
        
        Returns:
            Embedding dimension
        """
        return self.feature_dim

