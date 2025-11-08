"""
PyTorch Lightning module wrapper for MultiModalModel.

This module provides a LightningModule wrapper that enables training
with PyTorch Lightning, including metrics, checkpointing, and optimization.
"""
from typing import List, Optional, Dict, Any
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from configilm.metrics import get_classification_metric_collection
from configilm.extra.BENv2_utils import NEW_LABELS

from multimodal.model import MultiModalModel


class MultiModalLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module wrapper for MultiModalModel.
    
    This wrapper enables:
    - Automatic training/validation/test loops
    - Metrics computation
    - Checkpointing
    - Learning rate scheduling
    - Parameter grouping for different learning rates
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        lr: float = 1e-3,
        warmup: Optional[int] = None,
        dinov3_checkpoint: Optional[str] = None,
        resnet_checkpoint: Optional[str] = None,
        freeze_dinov3: bool = False,
        freeze_resnet: bool = False,
        dinov3_model_name: Optional[str] = None,
    ):
        """
        Initialize multimodal Lightning module.
        
        Args:
            config: Configuration dictionary for MultiModalModel
            lr: Learning rate
            warmup: Warmup steps (None for automatic calculation)
            dinov3_checkpoint: Path to checkpoint file to load DINOv3 backbone weights from
            resnet_checkpoint: Path to checkpoint file to load ResNet backbone weights from
            freeze_dinov3: Whether to freeze DINOv3 backbone after loading checkpoint
            freeze_resnet: Whether to freeze ResNet backbone after loading checkpoint
            dinov3_model_name: Explicit DINOv3 model name (if None, will be inferred from checkpoint or config)
        """
        super().__init__()
        self.lr = lr
        self.warmup = None if warmup is None or warmup < 0 else warmup
        self.config = config
        
        # Note: s2_band_order is not needed since the dataloader ensures correct channel order
        # based on channel_configurations registered in the training script
        
        # Infer DINOv3 model size from checkpoint if provided
        if dinov3_checkpoint is not None and dinov3_model_name is None:
            inferred_model_name = self._infer_dinov3_model_from_checkpoint(dinov3_checkpoint)
            if inferred_model_name:
                print(f"Inferred DINOv3 model from checkpoint: {inferred_model_name}")
                if config is None:
                    config = {}
                if "backbones" not in config:
                    config["backbones"] = {}
                if "dinov3" not in config["backbones"]:
                    config["backbones"]["dinov3"] = {}
                config["backbones"]["dinov3"]["model_name"] = inferred_model_name
        
        # Override with explicit model name if provided
        if dinov3_model_name is not None:
            if config is None:
                config = {}
            if "backbones" not in config:
                config["backbones"] = {}
            if "dinov3" not in config["backbones"]:
                config["backbones"]["dinov3"] = {}
            config["backbones"]["dinov3"]["model_name"] = dinov3_model_name
        
        # Create model
        self.model = MultiModalModel(config=config)
        
        # Load checkpoints if provided
        if dinov3_checkpoint is not None:
            self._load_backbone_from_checkpoint(
                checkpoint_path=dinov3_checkpoint,
                backbone_name="dinov3",
                freeze=freeze_dinov3,
            )
        
        if resnet_checkpoint is not None:
            self._load_backbone_from_checkpoint(
                checkpoint_path=resnet_checkpoint,
                backbone_name="resnet",
                freeze=freeze_resnet,
            )
        
        # Loss function
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        # Metrics
        num_classes = config.get("classifier", {}).get("num_classes", 19) if config else 19
        self.val_metrics_micro = get_classification_metric_collection(
            "multilabel", "micro", num_labels=num_classes, prefix="val/"
        )
        self.val_metrics_macro = get_classification_metric_collection(
            "multilabel", "macro", num_labels=num_classes, prefix="val/"
        )
        self.val_metrics_samples = get_classification_metric_collection(
            "multilabel", "sample", num_labels=num_classes, prefix="val/"
        )
        self.val_metrics_class = get_classification_metric_collection(
            "multilabel", None, num_labels=num_classes, prefix="val/"
        )
        self.test_metrics_micro = get_classification_metric_collection(
            "multilabel", "micro", num_labels=num_classes, prefix="test/"
        )
        self.test_metrics_macro = get_classification_metric_collection(
            "multilabel", "macro", num_labels=num_classes, prefix="test/"
        )
        self.test_metrics_samples = get_classification_metric_collection(
            "multilabel", "sample", num_labels=num_classes, prefix="test/"
        )
        self.test_metrics_class = get_classification_metric_collection(
            "multilabel", None, num_labels=num_classes, prefix="test/"
        )
        
        # Output lists for validation and test
        self.val_output_list: List[dict] = []
        self.test_output_list: List[dict] = []
    
    def _infer_dinov3_model_from_checkpoint(self, checkpoint_path: str) -> Optional[str]:
        """
        Infer DINOv3 model name from checkpoint path or checkpoint contents.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Inferred DINOv3 model name or None
        """
        from pathlib import Path
        
        # Strategy 1: Try to infer from filename
        ckpt_path = Path(checkpoint_path)
        filename = ckpt_path.name.lower()
        
        # Check filename for model size indicators
        if "dinov3-large" in filename or "dinov3-l" in filename or "dinov3l" in filename:
            return "facebook/dinov3-vitl16-pretrain-lvd1689m"
        elif "dinov3-base" in filename or "dinov3-b" in filename or "dinov3b" in filename:
            return "facebook/dinov3-vitb16-pretrain-lvd1689m"
        elif "dinov3-small" in filename or "dinov3-s" in filename or "dinov3s" in filename:
            return "facebook/dinov3-vits16-pretrain-lvd1689m"
        elif "dinov3-giant" in filename or "dinov3-g" in filename or "dinov3g" in filename:
            return "facebook/dinov3-vitg16-pretrain-lvd1689m"
        
        # Strategy 2: Try to load checkpoint and check hyperparameters
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Check hyperparameters
            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                if 'dinov3_model_name' in hparams:
                    return hparams['dinov3_model_name']
                if 'architecture' in hparams:
                    arch = hparams['architecture'].lower()
                    if 'large' in arch or 'l' in arch:
                        return "facebook/dinov3-vitl16-pretrain-lvd1689m"
                    elif 'base' in arch or 'b' in arch:
                        return "facebook/dinov3-vitb16-pretrain-lvd1689m"
                    elif 'small' in arch or 's' in arch:
                        return "facebook/dinov3-vits16-pretrain-lvd1689m"
                    elif 'giant' in arch or 'g' in arch:
                        return "facebook/dinov3-vitg16-pretrain-lvd1689m"
            
            # Strategy 3: Check state_dict to infer model size
            state_dict = checkpoint.get('state_dict', checkpoint)
            # Look for embedding dimension in keys
            for key in list(state_dict.keys())[:20]:
                if 'embeddings' in key.lower() and 'cls_token' in key.lower():
                    shape = state_dict[key].shape
                    if len(shape) >= 2:
                        hidden_size = shape[-1]
                        if hidden_size == 1024:
                            return "facebook/dinov3-vitl16-pretrain-lvd1689m"
                        elif hidden_size == 768:
                            return "facebook/dinov3-vitb16-pretrain-lvd1689m"
                        elif hidden_size == 384:
                            return "facebook/dinov3-vits16-pretrain-lvd1689m"
                        elif hidden_size == 1536:
                            return "facebook/dinov3-vitg16-pretrain-lvd1689m"
        except Exception as e:
            print(f"  Warning: Could not infer DINOv3 model from checkpoint: {e}")
        
        return None
    
    def _load_backbone_from_checkpoint(
        self,
        checkpoint_path: str,
        backbone_name: str,
        freeze: bool = False,
    ):
        """
        Load backbone weights from a checkpoint file.
        
        This function can load weights from:
        - BigEarthNetv2_0_ImageClassifier checkpoints (DINOv3 or ResNet)
        - Direct backbone state dicts
        
        Args:
            checkpoint_path: Path to checkpoint file
            backbone_name: Name of backbone ('dinov3' or 'resnet')
            freeze: Whether to freeze the backbone after loading
        """
        from pathlib import Path
        
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading {backbone_name} backbone from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Print first few keys for debugging
        print(f"  Checkpoint contains {len(state_dict)} keys")
        print(f"  Sample keys: {list(state_dict.keys())[:10]}")
        
        # Check for classification layers that will be skipped (for ResNet)
        if backbone_name == "resnet":
            classifier_keys = [k for k in state_dict.keys() if any(kw in k.lower() for kw in ["fc", "classifier", "head"])]
            if classifier_keys:
                print(f"  Will skip {len(classifier_keys)} classification layer keys: {classifier_keys[:5]}")
        
        # Map state dict keys to our model structure
        # The checkpoint from BigEarthNetv2_0_ImageClassifier has keys like:
        # - "model.backbone.*" for DINOv3
        # - "model.model.*" for ConfigILM/ResNet models
        mapped_state_dict = {}
        target_prefix = f"{backbone_name}_backbone"
        
        # Strategy 1: Try to find backbone keys directly
        # For DINOv3: look for "model.backbone.*" or keys containing "dinov3"
        # For ResNet: look for "model.model.*" or keys containing "resnet"
        patterns_to_try = []
        
        if backbone_name == "dinov3":
            # Look for DINOv3 backbone keys
            patterns_to_try = [
                ("model.backbone.", "backbone."),  # DINOv3Backbone structure
                ("model.model.backbone.", "backbone."),  # Nested
            ]
            
            # Also try to find any keys with "backbone" that might be DINOv3
            for key, value in state_dict.items():
                if "backbone" in key.lower() and ("dinov3" in key.lower() or "vit" in key.lower() or "transformer" in key.lower()):
                    # Map to our structure
                    if key.startswith("model."):
                        new_key = key.replace("model.", f"{target_prefix}.")
                    else:
                        new_key = f"{target_prefix}.{key}"
                    mapped_state_dict[new_key] = value
        
        elif backbone_name == "resnet":
            # Look for ResNet backbone keys
            # ResNet models in ConfigILM have structure: model.model.*
            # We need to skip the final classification layer (fc, classifier, head)
            patterns_to_try = [
                ("model.model.", "backbone."),  # ConfigILM ResNet structure
                ("model.", "backbone."),  # Direct model structure
            ]
            
            # Keys to skip (classification layers)
            skip_keywords = ["classifier", "head", "fc", "fc.", ".fc"]
            
            # Also try to find any keys with "resnet" or "backbone"
            for key, value in state_dict.items():
                # Skip classification layers
                if any(skip_kw in key.lower() for skip_kw in skip_keywords):
                    continue
                
                if ("resnet" in key.lower() or "backbone" in key.lower() or 
                    "model.model." in key or "model." in key):
                    # Map to our structure
                    if key.startswith("model.model."):
                        # ConfigILM structure: model.model.conv1 -> resnet_backbone.backbone.0.conv1
                        # We need to map model.model.* to resnet_backbone.backbone.*
                        # But skip fc layer
                        if ".fc" in key or "fc." in key:
                            continue
                        new_key = key.replace("model.model.", f"{target_prefix}.backbone.")
                    elif key.startswith("model."):
                        # Skip fc layer
                        if ".fc" in key or "fc." in key:
                            continue
                        new_key = key.replace("model.", f"{target_prefix}.")
                    else:
                        new_key = f"{target_prefix}.{key}"
                    mapped_state_dict[new_key] = value
        
        # Strategy 2: Try pattern-based mapping
        skip_keywords = ["classifier", "head", "fc", "fc.", ".fc"]
        for old_pattern, new_pattern in patterns_to_try:
            for key, value in state_dict.items():
                if old_pattern in key:
                    # Skip classification layers
                    if any(skip_kw in key.lower() for skip_kw in skip_keywords):
                        continue
                    new_key = key.replace(old_pattern, f"{target_prefix}.{new_pattern}")
                    if new_key not in mapped_state_dict:  # Avoid duplicates
                        mapped_state_dict[new_key] = value
        
        # Strategy 3: If still no keys found, try to load the entire model and extract backbone
        if len(mapped_state_dict) == 0:
            print(f"  Warning: No direct matches found, trying to extract from full model...")
            # Try to load as a full model checkpoint and extract backbone
            # This is a fallback - we'll try to match any keys that might work
            skip_keywords = ["classifier", "head", "fc", "fc.", ".fc"]
            for key, value in state_dict.items():
                # Skip classifier and other non-backbone parts
                if any(skip_kw in key.lower() for skip_kw in skip_keywords):
                    continue
                
                # Try to map to backbone structure
                if backbone_name == "dinov3" and ("backbone" in key.lower() or "embeddings" in key.lower() or "encoder" in key.lower()):
                    new_key = key.replace("model.", f"{target_prefix}.")
                    mapped_state_dict[new_key] = value
                elif backbone_name == "resnet":
                    # ResNet backbone layers: conv, bn, layer, avgpool (but NOT fc)
                    if ("backbone" in key.lower() or "conv" in key.lower() or 
                        "bn" in key.lower() or "layer" in key.lower() or 
                        "avgpool" in key.lower() or "maxpool" in key.lower()):
                        # Make sure it's not fc layer
                        if ".fc" not in key and "fc." not in key:
                            new_key = key.replace("model.model.", f"{target_prefix}.backbone.")
                            if new_key not in mapped_state_dict:
                                mapped_state_dict[new_key] = value
        
        if len(mapped_state_dict) > 0:
            # Load the mapped state dict
            print(f"  Attempting to load {len(mapped_state_dict)} parameters...")
            missing_keys, unexpected_keys = self.model.load_state_dict(mapped_state_dict, strict=False)
            
            if missing_keys:
                print(f"  Missing keys (not loaded): {len(missing_keys)}")
                if len(missing_keys) <= 10:
                    for key in missing_keys:
                        print(f"    - {key}")
            if unexpected_keys:
                print(f"  Unexpected keys (ignored): {len(unexpected_keys)}")
            
            loaded_count = len(mapped_state_dict) - len(missing_keys)
            print(f"  Successfully loaded {loaded_count} parameters")
            
            # Freeze if requested
            if freeze:
                if backbone_name == "dinov3":
                    for param in self.model.dinov3_backbone.parameters():
                        param.requires_grad = False
                    print(f"  Frozen {backbone_name} backbone parameters")
                elif backbone_name == "resnet":
                    for param in self.model.resnet_backbone.parameters():
                        param.requires_grad = False
                    print(f"  Frozen {backbone_name} backbone parameters")
        else:
            print(f"  Error: No matching parameters found for {backbone_name} backbone")
            print(f"  Available keys in checkpoint:")
            for key in list(state_dict.keys())[:20]:
                print(f"    - {key}")
            raise ValueError(f"Could not load {backbone_name} backbone from checkpoint. No matching keys found.")
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        # x is (B, C, H, W) where C = RGB (3) + S2_non-RGB (N) + S1 (2)
        # Based on channel_configurations, the order is: [RGB, S2_non-RGB, S1]
        # Split directly into RGB and (non-RGB + S1) as the model expects
        rgb_data = x[:, :3, :, :]  # RGB channels (indices 0-2)
        non_rgb_s1_data = x[:, 3:, :, :]  # S2 non-RGB + S1 (all remaining channels)
        
        logits = self.model(rgb_data, non_rgb_s1_data)
        loss = self.loss(logits, y)
        self.log("train/loss", loss)
        
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            current_gpu_mem_mb = torch.cuda.memory_allocated(current_gpu) / 1024 ** 2
            self.log("train/GPU_memory_MB", current_gpu_mem_mb)
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        rgb_data = x[:, :3, :, :]  # RGB channels (indices 0-2)
        non_rgb_s1_data = x[:, 3:, :, :]  # S2 non-RGB + S1 (all remaining channels)
        
        logits = self.model(rgb_data, non_rgb_s1_data)
        loss = self.loss(logits, y)
        self.val_output_list += [{"loss": loss, "outputs": logits, "labels": y}]
    
    def on_validation_epoch_start(self):
        """Reset validation output list at start of epoch."""
        super().on_validation_epoch_start()
        self.val_output_list = []
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at end of epoch."""
        avg_loss = torch.stack([x["loss"] for x in self.val_output_list]).mean()
        self.log("val/loss", avg_loss)
        
        preds = torch.cat([x["outputs"] for x in self.val_output_list])
        labels = torch.cat([x["labels"] for x in self.val_output_list]).long()
        
        metrics_macro = self.val_metrics_macro(preds, labels)
        self.log_dict(metrics_macro)
        self.val_metrics_macro.reset()
        
        metrics_micro = self.val_metrics_micro(preds, labels)
        self.log_dict(metrics_micro)
        self.val_metrics_micro.reset()
        
        metrics_samples = self.val_metrics_samples(preds.unsqueeze(-1), labels.unsqueeze(-1))
        metrics_samples = {k: v.mean() for k, v in metrics_samples.items()}
        self.log_dict(metrics_samples)
        self.val_metrics_samples.reset()
        
        class_names = NEW_LABELS
        metrics_class = self.val_metrics_class(preds, labels)
        classwise_acc = {
            f"val/ClasswiseAccuracy/{class_names[i]}": metrics_class["val/MultilabelAccuracy_class"][i]
            for i in range(len(class_names))
        }
        self.log_dict(classwise_acc)
        self.val_metrics_class.reset()
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        rgb_data = x[:, :3, :, :]  # RGB channels (indices 0-2)
        non_rgb_s1_data = x[:, 3:, :, :]  # S2 non-RGB + S1 (all remaining channels)
        
        logits = self.model(rgb_data, non_rgb_s1_data)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.test_output_list += [{"loss": loss, "outputs": logits, "labels": y}]
    
    def on_test_epoch_end(self):
        """Compute test metrics at end of epoch."""
        avg_loss = torch.stack([x["loss"] for x in self.test_output_list]).mean()
        self.log("test/loss", avg_loss)
        
        preds = torch.cat([x["outputs"] for x in self.test_output_list])
        labels = torch.cat([x["labels"] for x in self.test_output_list]).long()
        
        metrics_macro = self.test_metrics_macro(preds, labels)
        self.log_dict(metrics_macro)
        self.test_metrics_macro.reset()
        
        metrics_micro = self.test_metrics_micro(preds, labels)
        self.log_dict(metrics_micro)
        self.test_metrics_micro.reset()
        
        metrics_samples = self.test_metrics_samples(preds.unsqueeze(-1), labels.unsqueeze(-1))
        metrics_samples = {k: v.mean() for k, v in metrics_samples.items()}
        self.log_dict(metrics_samples)
        self.test_metrics_samples.reset()
        
        class_names = NEW_LABELS
        metrics_class = self.test_metrics_class(preds, labels)
        classwise_acc = {
            f"test/ClasswiseAccuracy/{class_names[i]}": metrics_class["test/MultilabelAccuracy_class"][i]
            for i in range(len(class_names))
        }
        self.log_dict(classwise_acc)
        self.test_metrics_class.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Use parameter groups from model if available
        param_groups = self.model.get_parameter_groups()
        
        # Override learning rates if specified in config
        if self.config and "backbones" in self.config:
            if "dinov3" in self.config["backbones"]:
                for group in param_groups:
                    if group["name"] == "dinov3_backbone":
                        group["lr"] = self.config["backbones"]["dinov3"].get("lr", self.lr)
            if "resnet101" in self.config["backbones"]:
                for group in param_groups:
                    if group["name"] == "resnet101_backbone":
                        group["lr"] = self.config["backbones"]["resnet101"].get("lr", self.lr)
        
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=0.01)
        
        # Calculate warmup steps
        max_intervals = int(
            self.trainer.max_epochs * len(self.trainer.datamodule.train_ds) / self.trainer.datamodule.batch_size
        )
        if self.warmup is not None:
            print(f"Using warmup: {self.warmup} steps")
            warmup = self.warmup
        else:
            warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0
        
        print(f"Optimizing for up to {max_intervals} steps with warmup for {warmup} steps")
        
        from configilm.extra.CustomTorchClasses import LinearWarmupCosineAnnealingLR
        lr_scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup,
                max_epochs=max_intervals,
                warmup_start_lr=self.lr / 10,
                eta_min=self.lr / 10,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
    
    def forward(self, rgb_data: torch.Tensor, non_rgb_s1_data: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(rgb_data, non_rgb_s1_data)

