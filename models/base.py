"""
Base classes for classifiers.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple


class BaseClassifier(nn.Module, ABC):
    """
    Base class for all classifiers.
    """
    
    def __init__(self, num_classes: int = 19):
        """
        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Logits of shape (B, num_classes)
        """
        pass


class BaseUnimodalClassifier(BaseClassifier):
    """
    Base class for unimodal classifiers that use a single backbone.
    """
    
    def __init__(
        self,
        num_classes: int = 19,
        backbone: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
    ):
        """
        Args:
            num_classes: Number of output classes
            backbone: Backbone model (feature extractor)
            classifier: Classification head
        """
        super().__init__(num_classes)
        self.backbone = backbone
        self.classifier = classifier
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone and classifier.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        if self.backbone is None:
            raise ValueError("Backbone not initialized")
        if self.classifier is None:
            raise ValueError("Classifier not initialized")
        
        features = self._extract_features(x)
        logits = self.classifier(features)
        return logits
    
    @abstractmethod
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            Features tensor
        """
        pass


class BaseMultimodalClassifier(BaseClassifier):
    """
    Base class for multimodal classifiers that fuse features from multiple backbones.
    """
    
    def __init__(
        self,
        num_classes: int = 19,
        backbones: Optional[List[nn.Module]] = None,
        fusion: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
    ):
        """
        Args:
            num_classes: Number of output classes
            backbones: List of backbone models
            fusion: Fusion module to combine features from multiple backbones
            classifier: Classification head
        """
        super().__init__(num_classes)
        self.backbones = nn.ModuleList(backbones) if backbones else nn.ModuleList()
        self.fusion = fusion
        self.classifier = classifier
    
    def forward(self, x: torch.Tensor, channel_ranges: Optional[List[Tuple[int, int]]] = None) -> torch.Tensor:
        """
        Forward pass through multiple backbones, fusion, and classifier.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C is the sum of all backbone input channels
            channel_ranges: List of (start, end) tuples indicating which channels each backbone should use.
                          If None, assumes sequential channel assignment.
            
        Returns:
            Logits of shape (B, num_classes)
        """
        if len(self.backbones) == 0:
            raise ValueError("No backbones initialized")
        if self.fusion is None:
            raise ValueError("Fusion module not initialized")
        if self.classifier is None:
            raise ValueError("Classifier not initialized")
        
        features = self._extract_features(x, channel_ranges)
        fused_features = self.fusion(features)
        logits = self.classifier(fused_features)
        return logits
    
    @abstractmethod
    def _extract_features(
        self,
        x: torch.Tensor,
        channel_ranges: Optional[List[Tuple[int, int]]] = None
    ) -> List[torch.Tensor]:
        """
        Extract features from all backbones.
        
        Args:
            x: Input tensor
            channel_ranges: List of (start, end) tuples for channel assignment
            
        Returns:
            List of feature tensors from each backbone
        """
        pass

