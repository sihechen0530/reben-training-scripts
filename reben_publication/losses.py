"""
Loss functions for multilabel classification.
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    This loss function addresses the class imbalance problem in multi-label classification
    by applying asymmetric focusing on negative samples and asymmetric clipping.
    
    Reference: https://arxiv.org/abs/2009.14119
    """
    
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples (default: 4)
            gamma_pos: Focusing parameter for positive samples (default: 1)
            clip: Clipping value for negative probabilities (default: 0.05)
            eps: Small epsilon value to prevent log(0) (default: 1e-8)
            disable_torch_grad_focal_loss: Whether to disable gradient computation for focal loss (default: True)
        """
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
    
    def forward(self, x, y):
        """
        Forward pass.
        
        Args:
            x: Input logits of shape (batch_size, num_classes)
            y: Targets (multi-label binarized vector) of shape (batch_size, num_classes)
        
        Returns:
            Loss value (scalar)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Basic Cross Entropy Calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        
        # Asymmetric Focusing
        # Note: focusing only on negatives!
        if self.disable_torch_grad_focal_loss:
            # Disable gradient computation for the focusing term to save memory
            with torch.no_grad():
                pt = xs_neg.detach()
            loss = -1 * (self.gamma_pos * los_pos + self.gamma_neg * los_neg * (pt ** self.gamma_neg))
        else:
            loss = -1 * (self.gamma_pos * los_pos + self.gamma_neg * los_neg * (xs_neg ** self.gamma_neg))
        
        return loss.sum()

