"""
DINOv3 Domain-Adaptive Pre-training (DAPT) for BigEarthNetv2.

This script implements unsupervised pre-training of DINOv3 on BigEarthNetv2
using the self-supervised DINO objective with multi-crop augmentation.

Key features:
- Weight Inflation: Adapts 3-channel DINOv3 to 14 channels (S1+S2)
- Multi-Crop Augmentation: 2 global crops (80%) + 8 local crops (30%)
- Student-Teacher Architecture: Momentum updates and centering
- Native Resolution: Trains at 128x128 (no upscaling)
"""

import os
import math
from typing import Optional, List, Tuple
import numpy as np

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
from configilm.extra.BENv2_utils import STANDARD_BANDS

# DINOv3 support
try:
    from reben_publication.DINOv3Backbone import DINOv3Backbone
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


class MultiCropAugmentation:
    """
    Multi-crop augmentation for DINOv3 self-supervised learning.
    
    Creates multiple views of the same image:
    - 2 global crops: Large crops covering ~80% of the image
    - 8 local crops: Small crops covering ~30% of the image
    """
    
    def __init__(
        self,
        global_crop_size: int = 128,
        local_crop_size: int = 96,
        global_crop_scale: Tuple[float, float] = (0.7, 1.0),
        local_crop_scale: Tuple[float, float] = (0.3, 0.7),
        num_local_crops: int = 8,
        num_global_crops: int = 2,
    ):
        """
        Args:
            global_crop_size: Size of global crops after resizing
            local_crop_size: Size of local crops after resizing
            global_crop_scale: Scale range for global crops (relative to image)
            local_crop_scale: Scale range for local crops (relative to image)
            num_local_crops: Number of local crops to generate
            num_global_crops: Number of global crops to generate
        """
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.num_local_crops = num_local_crops
        self.num_global_crops = num_global_crops
    
    def __call__(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply multi-crop augmentation to an image.
        
        Args:
            image: Input image tensor of shape (C, H, W)
            
        Returns:
            List of augmented crops: [global_crop_1, global_crop_2, local_crop_1, ..., local_crop_8]
        """
        crops = []
        
        # Generate global crops
        for _ in range(self.num_global_crops):
            crop = self._random_crop(
                image,
                crop_size=self.global_crop_size,
                scale=self.global_crop_scale,
            )
            crop = self._apply_augmentation(crop, is_global=True)
            crops.append(crop)
        
        # Generate local crops
        for _ in range(self.num_local_crops):
            crop = self._random_crop(
                image,
                crop_size=self.local_crop_size,
                scale=self.local_crop_scale,
            )
            crop = self._apply_augmentation(crop, is_global=False)
            crops.append(crop)
        
        return crops
    
    def _apply_augmentation(self, image: torch.Tensor, is_global: bool = True) -> torch.Tensor:
        """
        Apply augmentation to a crop (tensor-based, no PIL conversion).
        
        Args:
            image: Image tensor (C, H, W)
            is_global: Whether this is a global crop (stronger augmentation)
            
        Returns:
            Augmented image tensor
        """
        device = image.device
        
        # Random horizontal flip
        if torch.rand(1, device=device) < 0.5:
            image = torch.flip(image, dims=[2])
        
        # Random vertical flip
        if torch.rand(1, device=device) < 0.5:
            image = torch.flip(image, dims=[1])
        
        # Color jitter (for RGB channels only, if available)
        if image.shape[0] >= 3 and torch.rand(1, device=device) < (0.8 if is_global else 0.8):
            # Apply brightness, contrast, saturation to RGB channels
            rgb = image[:3]
            # Brightness
            brightness_factor = 1.0 + (torch.rand(1, device=device) - 0.5) * 0.4
            rgb = rgb * brightness_factor
            # Contrast
            contrast_factor = 1.0 + (torch.rand(1, device=device) - 0.5) * 0.4
            mean = rgb.mean(dim=(1, 2), keepdim=True)
            rgb = (rgb - mean) * contrast_factor + mean
            # Saturation (simplified)
            gray = rgb.mean(dim=0, keepdim=True)
            saturation_factor = 1.0 + (torch.rand(1, device=device) - 0.5) * 0.2
            rgb = gray + (rgb - gray) * saturation_factor
            image = torch.cat([rgb, image[3:]], dim=0)
        
        # Gaussian blur (if applicable)
        if torch.rand(1, device=device) < (1.0 if is_global else 0.5):
            # Simple blur using average pooling (approximation)
            kernel_size = 3
            padding = kernel_size // 2
            blurred = F.avg_pool2d(
                image.unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            ).squeeze(0)
            # Mix original and blurred
            mix_factor = 0.3
            image = image * (1 - mix_factor) + blurred * mix_factor
        
        # Clamp to valid range
        image = torch.clamp(image, 0.0, 1.0)
        
        return image
    
    def _random_crop(
        self,
        image: torch.Tensor,
        crop_size: int,
        scale: Tuple[float, float],
    ) -> torch.Tensor:
        """
        Random crop and resize to target size.
        
        Args:
            image: Input image (C, H, W)
            crop_size: Target crop size
            scale: Scale range (min, max) relative to image size
            
        Returns:
            Cropped and resized image (C, crop_size, crop_size)
        """
        _, h, w = image.shape
        
        # Calculate crop size based on scale
        scale_val = np.random.uniform(scale[0], scale[1])
        crop_h = int(h * scale_val)
        crop_w = int(w * scale_val)
        
        # Ensure crop doesn't exceed image size
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        
        # Random position
        top = np.random.randint(0, max(1, h - crop_h + 1))
        left = np.random.randint(0, max(1, w - crop_w + 1))
        
        # Crop
        crop = image[:, top:top+crop_h, left:left+crop_w]
        
        # Resize to target size
        crop = F.interpolate(
            crop.unsqueeze(0),
            size=(crop_size, crop_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return crop


class DINOLoss(nn.Module):
    """
    DINO self-supervised loss.
    
    Computes cross-entropy between student and teacher outputs with:
    - Temperature scaling
    - Sharpening
    - Centering
    """
    
    def __init__(
        self,
        out_dim: int = 65536,  # DINOv3 default
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 0,
        num_epochs: int = 100,
        num_local_crops: int = 8,
        num_global_crops: int = 2,
        center_momentum: float = 0.9,
    ):
        """
        Args:
            out_dim: Output dimension (number of prototypes)
            warmup_teacher_temp: Initial teacher temperature
            teacher_temp: Final teacher temperature
            warmup_teacher_temp_epochs: Number of epochs to warmup temperature
            num_epochs: Total number of epochs
            num_local_crops: Number of local crops
            num_global_crops: Number of global crops
            center_momentum: Momentum for centering
        """
        super().__init__()
        self.out_dim = out_dim
        self.num_local_crops = num_local_crops
        self.num_global_crops = num_global_crops
        self.center_momentum = center_momentum
        
        # Teacher temperature schedule
        self.warmup_teacher_temp = warmup_teacher_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.num_epochs = num_epochs
        
        # Register buffer for centering
        self.register_buffer("center", torch.zeros(1, out_dim))
        
        # Student temperature (fixed)
        self.student_temp = 0.1
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        batch_size: int,
        epoch: int = 0,
    ) -> torch.Tensor:
        """
        Compute DINO loss.
        
        Args:
            student_output: Student model outputs (B * num_crops, out_dim)
            teacher_output: Teacher model outputs (B * num_global_crops, out_dim)
            batch_size: Batch size
            epoch: Current epoch (for temperature scheduling)
            
        Returns:
            Loss value
        """
        # Compute teacher temperature (with warmup)
        teacher_temp = self._get_teacher_temp(epoch)
        
        num_total_crops = self.num_global_crops + self.num_local_crops
        
        # Validate shapes
        expected_student_size = batch_size * num_total_crops
        expected_teacher_size = batch_size * self.num_global_crops
        if student_output.shape[0] != expected_student_size:
            raise ValueError(
                f"Student output shape mismatch: got {student_output.shape[0]}, "
                f"expected {expected_student_size} (batch_size={batch_size}, num_total_crops={num_total_crops})"
            )
        if teacher_output.shape[0] != expected_teacher_size:
            raise ValueError(
                f"Teacher output shape mismatch: got {teacher_output.shape[0]}, "
                f"expected {expected_teacher_size} (batch_size={batch_size}, num_global_crops={self.num_global_crops})"
            )
        
        # Normalize and chunk student outputs
        student_out = student_output / self.student_temp
        # Reshape to (batch_size, num_total_crops, out_dim)
        student_out = student_out.view(batch_size, num_total_crops, -1)
        
        # Normalize and chunk teacher outputs
        teacher_out = teacher_output / teacher_temp
        # Reshape to (batch_size, num_global_crops, out_dim)
        teacher_out = teacher_out.view(batch_size, self.num_global_crops, -1)
        
        # Apply centering to teacher outputs
        teacher_out_softmax = F.softmax(teacher_out, dim=-1)  # (B, num_global, out_dim)
        # Update center: average over batch and global crops
        # Flatten to (B * num_global, out_dim) then take mean
        teacher_flat = teacher_out_softmax.view(-1, teacher_out_softmax.shape[-1])  # (B * num_global, out_dim)
        batch_center = teacher_flat.mean(dim=0, keepdim=True)  # (1, out_dim)
        # Update center (self.center is (1, out_dim))
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        # Subtract center from teacher outputs (broadcasting: (B, num_global, out_dim) - (1, out_dim))
        teacher_out = teacher_out_softmax - self.center  # (B, num_global, out_dim)
        
        # Compute loss: for each image, compare all student crops with all teacher crops
        # (except matching global crops)
        total_loss = 0
        n_loss_terms = 0
        
        for b in range(batch_size):
            # Teacher outputs for this image: (num_global, out_dim)
            teacher_b = teacher_out[b]  # (num_global, out_dim)
            
            # Student outputs for this image: (num_total_crops, out_dim)
            student_b = student_out[b]  # (num_total_crops, out_dim)
            
            # Compare each teacher crop with each student crop (except matching global)
            for t_idx in range(self.num_global_crops):
                t_out = teacher_b[t_idx:t_idx+1]  # (1, out_dim)
                
                for s_idx in range(num_total_crops):
                    # Skip matching global crop
                    if s_idx == t_idx and s_idx < self.num_global_crops:
                        continue
                    
                    s_out = student_b[s_idx:s_idx+1]  # (1, out_dim)
                    loss = torch.sum(-t_out * F.log_softmax(s_out, dim=-1), dim=-1)
                    total_loss += loss
                    n_loss_terms += 1
        
        return total_loss / max(n_loss_terms, 1)
    
    def _get_teacher_temp(self, epoch: int) -> float:
        """Get teacher temperature based on epoch (with warmup)."""
        if epoch < self.warmup_teacher_temp_epochs:
            return self.warmup_teacher_temp
        return self.teacher_temp
    
    @torch.no_grad()
    def _update_center(self, teacher_output: torch.Tensor):
        """Update center for centering mechanism."""
        # This is now handled in forward() method
        pass


class DINOv3Pretraining(pl.LightningModule):
    """
    PyTorch Lightning module for DINOv3 self-supervised pre-training.
    
    Implements student-teacher architecture with momentum updates.
    """
    
    def __init__(
        self,
        dinov3_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_input_channels: int = 14,  # S1 + S2
        image_size: int = 128,
        out_dim: int = 65536,  # DINOv3 default
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 30,
        num_local_crops: int = 8,
        num_global_crops: int = 2,
        global_crop_size: int = 128,
        local_crop_size: int = 96,
        momentum_teacher: float = 0.996,
        center_momentum: float = 0.9,
        lr: float = 5e-6,
        weight_decay: float = 0.04,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        freeze_last_layer: bool = True,
        freeze_last_layer_epochs: int = 1,
    ):
        """
        Args:
            dinov3_model_name: HuggingFace DINOv3 model name
            num_input_channels: Number of input channels (14 for S1+S2)
            image_size: Input image size
            out_dim: Output dimension (number of prototypes)
            warmup_teacher_temp: Initial teacher temperature
            teacher_temp: Final teacher temperature
            warmup_teacher_temp_epochs: Epochs to warmup temperature
            num_local_crops: Number of local crops
            num_global_crops: Number of global crops
            global_crop_size: Size of global crops
            local_crop_size: Size of local crops
            momentum_teacher: Momentum for teacher updates
            center_momentum: Momentum for centering
            lr: Learning rate
            weight_decay: Weight decay
            warmup_epochs: Number of warmup epochs
            max_epochs: Maximum number of epochs
            freeze_last_layer: Whether to freeze last layer initially
            freeze_last_layer_epochs: Epochs to freeze last layer
        """
        super().__init__()
        self.save_hyperparameters()
        
        if not DINOV3_AVAILABLE:
            raise ImportError(
                "DINOv3 not available. Install transformers: pip install transformers"
            )
        
        self.num_input_channels = num_input_channels
        self.image_size = image_size
        self.momentum_teacher = momentum_teacher
        self.freeze_last_layer = freeze_last_layer
        self.freeze_last_layer_epochs = freeze_last_layer_epochs
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        
        # Create student model (trainable)
        print(f"Creating student model: {dinov3_model_name}")
        self.student = self._create_backbone(
            dinov3_model_name,
            num_input_channels,
            image_size,
            out_dim,
        )
        
        # Create teacher model (momentum update)
        print(f"Creating teacher model: {dinov3_model_name}")
        self.teacher = self._create_backbone(
            dinov3_model_name,
            num_input_channels,
            image_size,
            out_dim,
        )
        
        # Initialize teacher with student weights
        self._init_teacher()
        
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Multi-crop augmentation
        self.multicrop = MultiCropAugmentation(
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            num_local_crops=num_local_crops,
            num_global_crops=num_global_crops,
        )
        
        # DINO loss
        self.criterion = DINOLoss(
            out_dim=out_dim,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            num_epochs=max_epochs,
            num_local_crops=num_local_crops,
            num_global_crops=num_global_crops,
            center_momentum=center_momentum,
        )
    
    def _create_backbone(
        self,
        model_name: str,
        num_input_channels: int,
        image_size: int,
        out_dim: int,
    ) -> nn.Module:
        """Create DINOv3 backbone with projection head."""
        # Create backbone (without classifier)
        backbone = DINOv3Backbone(
            model_name=model_name,
            num_classes=out_dim,  # Will be replaced by projection head
            num_input_channels=num_input_channels,
            image_size=image_size,
            pretrained=True,
        )
        
        # Store embed_dim for later use
        embed_dim = backbone.embed_dim
        
        # Determine projection head dimensions based on model size
        # For smaller models, use smaller hidden dimensions to reduce parameter count
        if embed_dim <= 384:  # Small model (vits16)
            hidden_dim = 1024
            # Use smaller out_dim if default is too large
            if out_dim > 16384:
                print(f"Warning: out_dim={out_dim} is very large for small model. "
                      f"Consider using 8192 or 16384 to reduce parameters.")
        elif embed_dim <= 768:  # Base model (vitb16)
            hidden_dim = 2048
        else:  # Large/Giant models
            hidden_dim = 2048
        
        # Replace classifier with projection head (MLP)
        # The projection head takes features from backbone and projects to out_dim
        backbone.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        
        # Calculate and print parameter count for projection head
        proj_params = sum(p.numel() for p in backbone.classifier.parameters())
        print(f"Projection head parameters: {proj_params:,} ({proj_params/1e6:.2f}M)")
        print(f"  Architecture: {embed_dim} -> {hidden_dim} -> {hidden_dim} -> {out_dim}")
        
        # Store embed_dim for access
        backbone.embed_dim = embed_dim
        
        return backbone
    
    def _init_teacher(self):
        """Initialize teacher with student weights."""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.copy_(s_param.data)
    
    @torch.no_grad()
    def _update_teacher(self):
        """Update teacher with momentum."""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data = t_param.data * self.momentum_teacher + s_param.data * (1 - self.momentum_teacher)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student model."""
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        """Training step with multi-crop augmentation."""
        # Extract images (ignore labels for unsupervised learning)
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_size = images.shape[0]
        
        # Apply multi-crop augmentation to each image in the batch
        all_crops = []
        for img in images:
            crops = self.multicrop(img)
            all_crops.extend(crops)
        
        # Resize all crops to the same size (global crop size) before stacking
        # This is necessary because global and local crops have different sizes
        target_size = self.multicrop.global_crop_size
        resized_crops = []
        for crop in all_crops:
            if crop.shape[1] != target_size or crop.shape[2] != target_size:
                # Resize to target size
                crop = F.interpolate(
                    crop.unsqueeze(0),
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            resized_crops.append(crop)
        
        # Stack all crops: (B * num_crops, C, H, W)
        all_crops = torch.stack(resized_crops)
        
        # Forward pass through student (all crops)
        student_output = self.student(all_crops)  # (B * num_crops, out_dim)
        
        # Forward pass through teacher (only global crops)
        with torch.no_grad():
            # Teacher only sees global crops
            num_global = self.multicrop.num_global_crops
            num_local = self.multicrop.num_local_crops
            num_total_crops = num_global + num_local
            
            # Extract global crops: first num_global crops for each image
            global_crops_list = []
            for i in range(batch_size):
                start_idx = i * num_total_crops
                end_idx = start_idx + num_global
                global_crops_list.append(all_crops[start_idx:end_idx])
            global_crops = torch.cat(global_crops_list, dim=0)  # (B * num_global, C, H, W)
            
            teacher_output = self.teacher(global_crops)  # (B * num_global, out_dim)
        
        # Compute loss
        loss = self.criterion(
            student_output,
            teacher_output,
            batch_size=batch_size,
            epoch=self.current_epoch,
        )
        
        # Update teacher with momentum
        self._update_teacher()
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.trainer.optimizers:
            self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            current_gpu_mem_mb = torch.cuda.memory_allocated(current_gpu) / 1024 ** 2
            self.log("train/GPU_memory_MB", current_gpu_mem_mb)
        
        return {"loss": loss}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Parameters to optimize
        params_groups = self._get_params_groups()
        
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            params_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Cosine annealing with warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.warmup_epochs * total_steps / self.max_epochs)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def _get_params_groups(self):
        """Get parameter groups for optimization."""
        regularized = []
        not_regularized = []
        
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            
            # Freeze last layer for first few epochs
            if self.freeze_last_layer and self.current_epoch < self.freeze_last_layer_epochs:
                # Freeze the last linear layer of the projection head
                if "classifier" in name:
                    parts = name.split(".")
                    if len(parts) >= 2:
                        try:
                            layer_idx = int(parts[-2]) if parts[-2].isdigit() else None
                            # Last layer is index 6 (0: LayerNorm, 1: Linear, 2: GELU, 3: LayerNorm, 4: Linear, 5: GELU, 6: LayerNorm, 7: Linear)
                            if layer_idx == 7 or (isinstance(parts[-2], str) and "7" in parts[-2]):
                                continue  # Skip last linear layer
                        except (ValueError, IndexError):
                            pass
            
            # Weight decay for weights, no decay for biases and normalization
            if len(param.shape) >= 2:
                regularized.append(param)
            else:
                not_regularized.append(param)
        
        return [
            {"params": regularized, "weight_decay": self.weight_decay},
            {"params": not_regularized, "weight_decay": 0.0},
        ]
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Update teacher momentum schedule
        # Increase momentum from 0.996 to 1.0 during training
        m = 1 - (1 - self.momentum_teacher) * (math.cos(math.pi * self.current_epoch / self.max_epochs) + 1) / 2
        self.momentum_teacher = m
        self.log("train/momentum_teacher", m)

