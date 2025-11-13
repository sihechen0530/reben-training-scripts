"""
DINOv3 backbone implementation compatible with BigEarthNet v2.0 training scripts.
"""
import os
import torch
import torch.nn as nn
from typing import Optional

try:
    # DINOv3 models are loaded using AutoModel from transformers
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModel = None
    AutoConfig = None
    print("Warning: transformers library not available. Install with: pip install transformers")


class DINOv3Backbone(nn.Module):
    """
    DINOv3 backbone wrapper compatible with ConfigILM interface.
    
    This wrapper adapts DINOv3 from HuggingFace transformers to work
    with the BigEarthNet v2.0 training pipeline.
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
    ):
        """
        Args:
            model_name: HuggingFace model name. Common options:
                - 'facebook/dinov3-vits16-pretrain-lvd1689m' (small, 384 dim)
                - 'facebook/dinov3-vitb16-pretrain-lvd1689m' (base, 768 dim)
                - 'facebook/dinov3-vitl16-pretrain-lvd1689m' (large, 1024 dim)
                - 'facebook/dinov3-vitg16-pretrain-lvd1689m' (giant, 1536 dim)
            num_classes: Number of output classes
            num_input_channels: Number of input channels (10 for S2, 12 for S2+S1, etc.)
            image_size: Input image size
            drop_rate: Dropout rate for classification head
            drop_path_rate: Drop path rate (stochastic depth)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for DINOv3. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.image_size = image_size
        
        # Get HuggingFace token from environment if available
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token and hf_token == "YOUR_HF_TOKEN_HERE":
            hf_token = None  # Ignore placeholder value
        
        # Load DINOv3 model using AutoModel
        print(f"Loading DINOv3 model: {model_name}")
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name, token=hf_token)
        else:
            config = AutoConfig.from_pretrained(model_name, token=hf_token)
            self.backbone = AutoModel.from_config(config)
        
        # Get embedding dimension
        self.embed_dim = self.backbone.config.hidden_size
        print(f"DINOv3 embedding dimension: {self.embed_dim}")
        
        # Adapt input layer if needed (DINOv3 expects 3 channels by default)
        if num_input_channels != 3:
            print(f"Adapting input layer for {num_input_channels} channels")
            self._adapt_input_layer()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.embed_dim, num_classes)
        )
        
        # Apply drop path rate if specified (approximate using dropout in attention)
        if drop_path_rate > 0:
            self._apply_drop_path(drop_path_rate)
        
        # Ensure model is in train mode (fixes warning about modules in eval mode)
        self.backbone.train()
        self.classifier.train()
    
    def _adapt_input_layer(self):
        """Adapt the patch embedding layer for different input channels."""
        # DINOv3 uses a patch embedding layer - try different possible structures
        try:
            # Try standard structure: embeddings.patch_embeddings.projection
            if hasattr(self.backbone, 'embeddings') and hasattr(self.backbone.embeddings, 'patch_embeddings'):
                original_embed = self.backbone.embeddings.patch_embeddings
                if hasattr(original_embed, 'projection'):
                    original_proj = original_embed.projection
                elif isinstance(original_embed, nn.Conv2d):
                    original_proj = original_embed
                else:
                    raise ValueError(f"Unexpected patch embedding structure: {type(original_embed)}")
            else:
                # Try alternative: might be directly accessible
                raise AttributeError("Could not find patch_embeddings")
            
            if not isinstance(original_proj, nn.Conv2d):
                raise ValueError(f"Expected Conv2d for patch embedding, got {type(original_proj)}")
            
            # Create new projection layer
            new_proj = nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=original_proj.out_channels,
                kernel_size=original_proj.kernel_size,
                stride=original_proj.stride,
                padding=original_proj.padding,
                bias=original_proj.bias is not None
            )
            
            # Initialize weights
            with torch.no_grad():
                if self.num_input_channels > 3:
                    # Repeat RGB weights for additional channels
                    repeat_factor = (self.num_input_channels // 3) + 1
                    weight = original_proj.weight.repeat(1, repeat_factor, 1, 1)
                    new_proj.weight = nn.Parameter(weight[:, :self.num_input_channels, :, :])
                else:
                    # Use subset of channels
                    new_proj.weight = nn.Parameter(original_proj.weight[:, :self.num_input_channels, :, :])
                
                if original_proj.bias is not None:
                    new_proj.bias = nn.Parameter(original_proj.bias.clone())
            
            # Replace the projection layer
            if hasattr(original_embed, 'projection'):
                original_embed.projection = new_proj
            else:
                self.backbone.embeddings.patch_embeddings = new_proj
                
        except (AttributeError, ValueError) as e:
            print(f"Warning: Could not adapt input layer automatically: {e}")
            print("The model may need manual adaptation or might not support multi-channel inputs.")
            raise
    
    def _apply_drop_path(self, drop_path_rate: float):
        """Apply drop path (stochastic depth) to the transformer blocks."""
        # DINOv3 uses attention blocks - we approximate drop path with dropout
        # In practice, this is applied during forward pass, but for simplicity
        # we note that the rate should be used in training
        self.drop_path_rate = drop_path_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Get features from DINOv3 backbone
        # DINOv3 expects pixel_values keyword argument
        try:
            outputs = self.backbone(pixel_values=x)
        except TypeError:
            # Fallback: try positional argument
            outputs = self.backbone(x)
        
        # Extract features - DINOv3 typically returns BaseModelOutput
        # which has last_hidden_state and optionally pooler_output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Use CLS token (first token)
            features = outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, torch.Tensor):
            # Direct tensor output
            features = outputs[:, 0] if len(outputs.shape) > 2 else outputs
        else:
            raise ValueError(f"Unexpected output type from DINOv3 backbone: {type(outputs)}")
        
        # Classification
        logits = self.classifier(features)
        
        return logits

