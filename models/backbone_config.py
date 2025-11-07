"""
Backbone configuration classes.
"""
from typing import Optional
from dataclasses import dataclass


@dataclass
class BackboneConfig:
    """Configuration for a single backbone."""
    name: str
    input_channels: int
    architecture: str = "resnet18"
    pretrained: bool = True
    dinov3_model_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {self.input_channels}")

