"""
DINOv3 backbone implementation for RGB image processing.

This module provides a DINOv3 backbone wrapper that can be used with
pretrained weights from HuggingFace transformers.
"""
import torch
import torch.nn as nn
from typing import Optional

try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModel = None
    AutoConfig = None


class DINOv3Backbone(nn.Module):
    """
    DINOv3 backbone for processing RGB images (3 channels).
    
    This backbone uses pretrained DINOv3 models from HuggingFace and
    can be frozen or fine-tuned during training.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        pretrained: bool = True,
        freeze: bool = False,
        input_channels: int = 3,
        image_size: int = 120,
    ):
        """
        Initialize DINOv3 backbone.
        
        Args:
            model_name: HuggingFace model name. Common options:
                - 'facebook/dinov3-vits16-pretrain-lvd1689m' (small, 384 dim)
                - 'facebook/dinov3-vitb16-pretrain-lvd1689m' (base, 768 dim)
                - 'facebook/dinov3-vitl16-pretrain-lvd1689m' (large, 1024 dim)
                - 'facebook/dinov3-vitg16-pretrain-lvd1689m' (giant, 1536 dim)
            pretrained: Whether to load pretrained weights
            freeze: Whether to freeze backbone parameters
            input_channels: Number of input channels (should be 3 for RGB)
            image_size: Input image size (for compatibility, not strictly used by DINOv3)
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for DINOv3. "
                "Install with: pip install transformers"
            )
        
        if input_channels != 3:
            raise ValueError(
                f"DINOv3 expects 3 input channels (RGB), got {input_channels}"
            )
        
        self.model_name = model_name
        self.input_channels = input_channels
        self.image_size = image_size
        self.freeze = freeze
        
        # Load DINOv3 model
        print(f"Loading DINOv3 model: {model_name} (pretrained={pretrained})")
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)
        
        # Get embedding dimension
        self.embed_dim = self.backbone.config.hidden_size
        print(f"DINOv3 embedding dimension: {self.embed_dim}")
        
        # Freeze parameters if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("DINOv3 backbone parameters frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv3 backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W) - RGB image
            
        Returns:
            Feature embeddings of shape (B, embed_dim)
        """
        if x.shape[1] != 3:
            raise ValueError(
                f"Expected 3 input channels (RGB), got {x.shape[1]}"
            )
        
        # DINOv3 expects pixel_values as input
        # The model returns a BaseModelOutputWithPooling object
        outputs = self.backbone(pixel_values=x)
        
        # Extract the CLS token embedding (pooler_output or first token)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Use CLS token (first token)
            features = outputs.last_hidden_state[:, 0]
        else:
            # Fallback: use the output directly
            features = outputs[0] if isinstance(outputs, tuple) else outputs
            if len(features.shape) > 2:
                features = features[:, 0]  # Take CLS token
        
        return features
    
    def get_embed_dim(self) -> int:
        """
        Get the embedding dimension of this backbone.
        
        Returns:
            Embedding dimension
        """
        return self.embed_dim

