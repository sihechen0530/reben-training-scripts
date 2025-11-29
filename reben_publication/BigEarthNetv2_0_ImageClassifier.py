"""
This is a script for supervised image classification using the BigEarthNet v2.0 dataset.
"""
# import packages
from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BENv2_utils import NEW_LABELS
from configilm.extra.CustomTorchClasses import LinearWarmupCosineAnnealingLR
from configilm.metrics import get_classification_metric_collection
from huggingface_hub import PyTorchModelHubMixin

# DINOv3 support
try:
    from reben_publication.DINOv3Backbone import DINOv3Backbone
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def class_aware_copy_paste(
    x: torch.Tensor,
    y: torch.Tensor,
    rare_class_indices: List[int],
    common_class_indices: List[int],
    prob: float = 0.5,
    min_scale: float = 0.2,
    max_scale: float = 0.5,
) -> tuple:
    """
    Apply Class-Aware Copy-Paste augmentation to a batch.
    
    Strategy:
    - Background: Sample an image with common classes
    - Foreground: Sample an image with rare classes
    - Action: Crop a random chunk from the rare image and paste it onto the common image
    - Label: Update the label to include both sets of classes
    
    Args:
        x: Input images tensor of shape (B, C, H, W)
        y: Labels tensor of shape (B, num_classes) for multilabel
        rare_class_indices: List of class indices that are rare
        common_class_indices: List of class indices that are common
        prob: Probability of applying copy-paste to each sample
        min_scale: Minimum scale of the cropped region (relative to image size)
        max_scale: Maximum scale of the cropped region (relative to image size)
    
    Returns:
        Augmented images and labels
    """
    batch_size = x.size(0)
    device = x.device
    H, W = x.size(2), x.size(3)
    
    # Create output tensors
    x_aug = x.clone()
    y_aug = y.clone()
    
    # For each sample, decide whether to apply copy-paste
    for i in range(batch_size):
        if torch.rand(1).item() > prob:
            continue  # Skip this sample
        
        # Find samples with rare classes (foreground) and common classes (background)
        # Background: current sample should have common classes
        has_common = torch.any(y[i, common_class_indices] > 0.5)
        if not has_common:
            continue  # Skip if current sample doesn't have common classes
        
        # Foreground: find another sample with rare classes
        rare_mask = torch.any(y[:, rare_class_indices] > 0.5, dim=1)
        rare_indices = torch.where(rare_mask)[0]
        
        if len(rare_indices) == 0:
            continue  # No samples with rare classes in this batch
        
        # Exclude current sample from rare candidates
        rare_indices = rare_indices[rare_indices != i]
        if len(rare_indices) == 0:
            continue
        
        # Randomly select a foreground sample
        fg_idx = rare_indices[torch.randint(len(rare_indices), (1,)).item()].item()
        
        # Generate random bounding box for the crop/paste region
        scale = torch.rand(1).item() * (max_scale - min_scale) + min_scale
        crop_w = int(W * scale)
        crop_h = int(H * scale)
        
        # Random position for the crop in foreground image
        fg_x1 = torch.randint(0, max(1, W - crop_w + 1), (1,)).item()
        fg_y1 = torch.randint(0, max(1, H - crop_h + 1), (1,)).item()
        fg_x2 = fg_x1 + crop_w
        fg_y2 = fg_y1 + crop_h
        
        # Random position for paste in background image
        bg_x1 = torch.randint(0, max(1, W - crop_w + 1), (1,)).item()
        bg_y1 = torch.randint(0, max(1, H - crop_h + 1), (1,)).item()
        bg_x2 = bg_x1 + crop_w
        bg_y2 = bg_y1 + crop_h
        
        # Copy region from foreground and paste onto background
        x_aug[i, :, bg_y1:bg_y2, bg_x1:bg_x2] = x[fg_idx, :, fg_y1:fg_y2, fg_x1:fg_x2]
        
        # Update labels: combine background and foreground labels
        y_aug[i] = torch.clamp(y[i] + y[fg_idx], 0.0, 1.0)
    
    return x_aug, y_aug


class BigEarthNetv2_0_ImageClassifier(pl.LightningModule, PyTorchModelHubMixin):
    """
    Wrapper around a pytorch module, allowing this module to be used in automatic
    training with pytorch lightning.
    Among other things, the wrapper allows us to do automatic training and removes the
    need to manage data on different devices (e.g. GPU and CPU).
    Also uses the PyTorchModelHubMixin to allow for easy saving and loading of the model to the Huggingface Hub.
    """

    def __init__(
            self,
            config: ILMConfiguration,
            lr: float = 1e-3,
            warmup: Optional[int] = None,
            dinov3_model_name: Optional[str] = None,
            linear_probe: bool = False,
            head_type: str = "linear",
            mlp_hidden_dims: Optional[List[int]] = None,
            head_dropout: Optional[float] = None,
            use_copy_paste: bool = False,
            copy_paste_prob: float = 0.3,
            copy_paste_min_scale: float = 0.5,
            copy_paste_max_scale: float = 0.7,
            rare_class_indices: Optional[List[int]] = None,
            common_class_indices: Optional[List[int]] = None,
            **kwargs,  # Accept extra kwargs for backward compatibility
    ):
        super().__init__()
        self.lr = lr
        self.warmup = None if warmup is None or warmup < 0 else warmup
        self.config = config
        self.linear_probe = linear_probe
        self.head_type = head_type.lower()
        self.mlp_hidden_dims = mlp_hidden_dims or []
        self.head_dropout = head_dropout if head_dropout is not None else getattr(config, 'drop_rate', 0.15)
        
        # Class-Aware Copy-Paste augmentation settings
        self.use_copy_paste = use_copy_paste
        self.copy_paste_prob = copy_paste_prob
        self.copy_paste_min_scale = copy_paste_min_scale
        self.copy_paste_max_scale = copy_paste_max_scale
        self.rare_class_indices = rare_class_indices or []
        self.common_class_indices = common_class_indices or []
        
        if self.use_copy_paste and (len(self.rare_class_indices) == 0 or len(self.common_class_indices) == 0):
            print("Warning: Copy-Paste enabled but rare/common class indices not provided. "
                  "Will be computed from datamodule during training setup.")
        
        # DEBUG: Print classifier configuration
        print(f"\n{'='*60}")
        print(f"DEBUG: BigEarthNetv2_0_ImageClassifier initialization")
        print(f"  linear_probe: {self.linear_probe}")
        print(f"  head_type: {self.head_type}")
        print(f"  mlp_hidden_dims: {self.mlp_hidden_dims}")
        print(f"  head_dropout: {self.head_dropout}")
        if self.use_copy_paste:
            print(f"  Using Class-Aware Copy-Paste augmentation")
            print(f"    prob={self.copy_paste_prob}, scale=[{self.copy_paste_min_scale}, {self.copy_paste_max_scale}]")
            if len(self.rare_class_indices) > 0:
                print(f"    rare_classes={len(self.rare_class_indices)}, common_classes={len(self.common_class_indices)}")
        print(f"{'='*60}\n")
        
        assert config.network_type == ILMType.IMAGE_CLASSIFICATION
        assert config.classes == 19
        
        # Check if DINOv3 is requested
        use_dinov3 = (
            DINOV3_AVAILABLE and 
            (dinov3_model_name is not None or
             (hasattr(config, 'timm_model_name') and 
              config.timm_model_name is not None and
              config.timm_model_name.startswith('dinov3')))
        )
        
        if use_dinov3:
            if not DINOV3_AVAILABLE:
                raise ImportError(
                    "DINOv3 requested but not available. "
                    "Install transformers: pip install transformers"
                )
            
            # Determine DINOv3 model name
            if dinov3_model_name is None:
                # Try to infer from timm_model_name
                # DINOv3 uses specific naming: facebook/dinov3-vit{s|b|l|g}16-pretrain-lvd1689m
                timm_name = getattr(config, 'timm_model_name', 'dinov3')
                if 'small' in timm_name.lower() or 's' in timm_name.lower():
                    dinov3_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
                elif 'base' in timm_name.lower() or 'b' in timm_name.lower():
                    dinov3_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
                elif 'large' in timm_name.lower() or 'l' in timm_name.lower():
                    dinov3_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
                elif 'giant' in timm_name.lower() or 'g' in timm_name.lower():
                    dinov3_model_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
                else:
                    # Default to small
                    dinov3_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
            
            print(f"Using DINOv3 backbone: {dinov3_model_name}")
            self.model = DINOv3Backbone(
                model_name=dinov3_model_name,
                num_classes=config.classes,
                num_input_channels=config.channels,
                image_size=config.image_size,
                drop_rate=getattr(config, 'drop_rate', 0.15),
                drop_path_rate=getattr(config, 'drop_path_rate', 0.0),
                pretrained=True,
            )
            self._override_classifier_head()
            # If linear probing, freeze backbone parameters
            if self.linear_probe:
                for p in self.model.backbone.parameters():
                    p.requires_grad = False
                # Ensure classifier remains trainable
                for p in self.model.classifier.parameters():
                    p.requires_grad = True
                # Keep the input adaptation layer (patch embedding projection) trainable
                try:
                    proj = None
                    if hasattr(self.model.backbone, 'embeddings') and hasattr(self.model.backbone.embeddings, 'patch_embeddings'):
                        pe = self.model.backbone.embeddings.patch_embeddings
                        if hasattr(pe, 'projection'):
                            proj = pe.projection
                        elif isinstance(pe, torch.nn.Conv2d):
                            proj = pe
                    # Unfreeze projection parameters if found
                    if proj is not None:
                        for p in proj.parameters():
                            p.requires_grad = True
                except Exception:
                    # If adaptation layer is not found, silently continue
                    pass
        else:
            # Use standard ConfigILM with timm models
            self.model = ConfigILM.ConfigILM(config)
        self.val_output_list: List[dict] = []
        self.test_output_list: List[dict] = []
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.val_metrics_micro = get_classification_metric_collection(
            "multilabel", "micro", num_labels=config.classes, prefix="val/"
        )
        self.val_metrics_macro = get_classification_metric_collection(
            "multilabel", "macro", num_labels=config.classes, prefix="val/"
        )
        self.val_metrics_samples = get_classification_metric_collection(
            "multilabel", "sample", num_labels=config.classes, prefix="val/"
        )
        self.val_metrics_class = get_classification_metric_collection(
            "multilabel", None, num_labels=config.classes, prefix="val/"
        )
        self.test_metrics_micro = get_classification_metric_collection(
            "multilabel", "micro", num_labels=config.classes, prefix="test/"
        )
        self.test_metrics_macro = get_classification_metric_collection(
            "multilabel", "macro", num_labels=config.classes, prefix="test/"
        )
        self.test_metrics_samples = get_classification_metric_collection(
            "multilabel", "sample", num_labels=config.classes, prefix="test/"
        )
        self.test_metrics_class = get_classification_metric_collection(
            "multilabel", None, num_labels=config.classes, prefix="test/"
        )

    def _override_classifier_head(self):
        """
        Replace the default classifier with either the original linear head
        or a configurable MLP head.
        """
        print(f"\nDEBUG: _override_classifier_head() called")
        print(f"  self.head_type = {self.head_type}")
        print(f"  self.mlp_hidden_dims = {self.mlp_hidden_dims}")
        
        if not hasattr(self.model, "classifier"):
            print(f"  WARNING: model has no 'classifier' attribute, skipping override")
            return

        if self.head_type not in {"linear", "mlp"}:
            raise ValueError(f"Unsupported head_type: {self.head_type}. Choose from ['linear', 'mlp'].")

        hidden_dims = []
        if self.head_type == "mlp":
            hidden_dims = self.mlp_hidden_dims or [1024, 512]
            print(f"  MLP mode: hidden_dims = {hidden_dims}")
        else:
            print(f"  Linear mode: no hidden dims")

        embed_dim = getattr(self.model, "embed_dim", None)
        print(f"  embed_dim = {embed_dim}")
        embed_dim = getattr(self.model, "embed_dim", None)
        print(f"  embed_dim = {embed_dim}")
        if embed_dim is None:
            # Fallback to classifier input dimension
            try:
                sample_layer = next(self.model.classifier.modules())
                embed_dim = sample_layer.normalized_shape[0] if isinstance(sample_layer, nn.LayerNorm) else None
            except StopIteration:
                embed_dim = None

        if embed_dim is None:
            print("  WARNING: Could not determine embed_dim for classifier override; keeping default head.")
            return

        layers: List[nn.Module] = [nn.LayerNorm(embed_dim)]
        in_dim = embed_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Dropout(self.head_dropout))
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim

        layers.append(nn.Dropout(self.head_dropout))
        layers.append(nn.Linear(in_dim, self.config.classes))
        self.model.classifier = nn.Sequential(*layers)
        
        print(f"  Classifier layers: {len(layers)}")
        print(f"  Final classifier structure:")
        for i, layer in enumerate(self.model.classifier):
            print(f"    [{i}] {layer}")
        print()

    def on_train_start(self):
        """Compute class frequencies if copy-paste is enabled and indices not provided."""
        super().on_train_start()
        if self.use_copy_paste and (len(self.rare_class_indices) == 0 or len(self.common_class_indices) == 0):
            from scripts.utils import compute_class_frequencies
            print("Computing class frequencies for Copy-Paste augmentation...")
            # Default threshold if not provided
            rare_threshold = getattr(self, 'copy_paste_rare_threshold', 0.1)
            _, rare_indices, common_indices = compute_class_frequencies(
                self.trainer.datamodule,
                num_classes=self.config.classes,
                max_samples=50000,  # Use subset for faster computation
                rare_threshold=rare_threshold
            )
            self.rare_class_indices = rare_indices
            self.common_class_indices = common_indices
            print(f"Copy-Paste: rare_classes={len(self.rare_class_indices)}, common_classes={len(self.common_class_indices)}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Apply Class-Aware Copy-Paste augmentation if enabled
        if self.use_copy_paste and len(self.rare_class_indices) > 0 and len(self.common_class_indices) > 0:
            x, y = class_aware_copy_paste(
                x, y,
                rare_class_indices=self.rare_class_indices,
                common_class_indices=self.common_class_indices,
                prob=self.copy_paste_prob,
                min_scale=self.copy_paste_min_scale,
                max_scale=self.copy_paste_max_scale,
            )
        
        x_hat = self.model(x)
        loss = self.loss(x_hat, y)
        self.log("train/loss", loss)
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            current_gpu_mem_mb = torch.cuda.memory_allocated(current_gpu) / 1024 ** 2
            self.log("train/GPU_memory_MB", current_gpu_mem_mb)
        return {"loss": loss}

    def configure_optimizers(self):
        # If linear probing with DINOv3, optimize classifier head and the input adaptation layer
        if hasattr(self, "model") and hasattr(self.model, "backbone") and hasattr(self.model, "classifier") and self.linear_probe:
            param_groups = [
                {"params": list(self.model.classifier.parameters())}
            ]
            # Add patch embedding projection if present
            try:
                proj = None
                if hasattr(self.model.backbone, 'embeddings') and hasattr(self.model.backbone.embeddings, 'patch_embeddings'):
                    pe = self.model.backbone.embeddings.patch_embeddings
                    if hasattr(pe, 'projection'):
                        proj = pe.projection
                    elif isinstance(pe, torch.nn.Conv2d):
                        proj = pe
                if proj is not None:
                    proj_params = [p for p in proj.parameters() if p.requires_grad]
                    if len(proj_params) > 0:
                        param_groups.append({"params": proj_params})
            except Exception:
                pass
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        # these are steps if interval is set to step
        max_intervals = int(
            self.trainer.max_epochs * len(self.trainer.datamodule.train_ds) / self.trainer.datamodule.batch_size
        )
        if self.warmup is not None:
            print(f"Overwriting warmup with {self.warmup}")
            warmup = self.warmup
        else:
            warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0

        print(f"Optimizing for up to {max_intervals} steps with warmup for {warmup} steps")

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

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, y)
        self.val_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []

    def on_validation_epoch_end(self):

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

        # get class names from datamodule
        class_names = NEW_LABELS
        metrics_class = self.val_metrics_class(preds, labels)
        classwise_acc = {
            f"val/ClasswiseAccuracy/{class_names[i]}": metrics_class["val/MultilabelAccuracy_class"][i]
            for i in range(len(class_names))
        }
        self.log_dict(classwise_acc)
        self.val_metrics_class.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.test_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_test_epoch_end(self):
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

    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)
