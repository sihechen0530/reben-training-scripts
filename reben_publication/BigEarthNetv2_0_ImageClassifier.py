"""
This is a script for supervised image classification using the BigEarthNet v2.0 dataset.
"""
# import packages
from typing import List, Optional
import numpy as np

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
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


def mixup_data(x, y, alpha=0.8):
    """
    Apply Mixup augmentation to a batch.
    
    Args:
        x: Input images (B, C, H, W)
        y: Labels (B, num_classes) - multi-label format
        alpha: Beta distribution parameter for mixing coefficient
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels for both images
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation to a batch.
    
    Args:
        x: Input images (B, C, H, W)
        y: Labels (B, num_classes) - multi-label format
        alpha: Beta distribution parameter for mixing coefficient
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels for both images
        lam: Mixing coefficient (adjusted for area)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get image dimensions
    _, _, H, W = x.size()
    
    # Generate random bounding box
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Clamp bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to match actual pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for Mixup/CutMix augmented batch.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a, y_b: Labels for both images
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
            use_weighted_sampler: bool = False,
            use_mixup: bool = False,
            mixup_alpha: float = 0.8,
            use_cutmix: bool = False,
            cutmix_alpha: float = 1.0,
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
        
        # Physical oversampling with Mixup/CutMix settings
        self.use_weighted_sampler = use_weighted_sampler
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        
        # Store sampler and dataloader for later use
        self._train_sampler = None
        self._train_dataloader = None
        
        # DEBUG: Print classifier configuration
        print(f"\n{'='*60}")
        print(f"DEBUG: BigEarthNetv2_0_ImageClassifier initialization")
        print(f"  linear_probe: {self.linear_probe}")
        print(f"  head_type: {self.head_type}")
        print(f"  mlp_hidden_dims: {self.mlp_hidden_dims}")
        print(f"  head_dropout: {self.head_dropout}")
        print(f"  use_weighted_sampler: {self.use_weighted_sampler}")
        print(f"  use_mixup: {self.use_mixup} (alpha={self.mixup_alpha})")
        print(f"  use_cutmix: {self.use_cutmix} (alpha={self.cutmix_alpha})")
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Apply Mixup or CutMix augmentation if enabled
        if self.use_mixup and self.training:
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=self.mixup_alpha)
            x_hat = self.model(mixed_x)
            loss = mixup_criterion(self.loss, x_hat, y_a, y_b, lam)
        elif self.use_cutmix and self.training:
            mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=self.cutmix_alpha)
            x_hat = self.model(mixed_x)
            loss = mixup_criterion(self.loss, x_hat, y_a, y_b, lam)
        else:
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
    
    def train_dataloader(self):
        """
        Override train_dataloader to use WeightedRandomSampler when enabled.
        This ensures rare classes appear more frequently in batches.
        """
        if not self.use_weighted_sampler:
            # Use default dataloader from datamodule
            return self.trainer.datamodule.train_dataloader()
        
        # Return cached dataloader if already computed
        if self._train_dataloader is not None:
            return self._train_dataloader
        
        # Calculate class weights based on training dataset
        train_ds = self.trainer.datamodule.train_ds
        batch_size = self.trainer.datamodule.batch_size
        num_workers = getattr(self.trainer.datamodule, 'num_workers_dataloader', 8)
        
        # Compute class frequencies
        print(f"\n{'='*60}")
        print("Computing class frequencies for WeightedRandomSampler...")
        print(f"{'='*60}")
        
        class_counts = torch.zeros(self.config.classes)
        total_samples = len(train_ds)
        
        # Sample a subset to estimate class frequencies (for efficiency)
        # If dataset is large, sample up to 10k samples
        sample_size = min(10000, total_samples)
        indices = torch.randperm(total_samples)[:sample_size]
        
        for idx in indices:
            try:
                _, label = train_ds[int(idx)]
                if isinstance(label, torch.Tensor):
                    class_counts += label.float()
                else:
                    # Convert to tensor if needed
                    label_tensor = torch.tensor(label, dtype=torch.float32)
                    class_counts += label_tensor
            except Exception as e:
                # Skip problematic samples
                continue
        
        # Compute class frequencies
        class_freqs = class_counts / (class_counts.sum() + 1e-8)
        
        # Compute inverse frequency weights (rare classes get higher weights)
        # Add small epsilon to avoid division by zero
        class_weights = 1.0 / (class_freqs + 1e-8)
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        
        print(f"Class frequencies (from {sample_size} samples):")
        for i, (name, freq, weight) in enumerate(zip(NEW_LABELS, class_freqs, class_weights)):
            print(f"  {i:2d}. {name:30s}: freq={freq:.4f}, weight={weight:.2f}")
        print(f"{'='*60}\n")
        
        # Compute sample weights: weight each sample by the maximum weight of its positive classes
        # This ensures samples with rare classes are sampled more frequently
        print("Computing sample weights...")
        sample_weights = torch.zeros(total_samples)
        
        # Process in batches to avoid memory issues
        batch_size_compute = 1000
        for i in range(0, total_samples, batch_size_compute):
            end_idx = min(i + batch_size_compute, total_samples)
            batch_indices = list(range(i, end_idx))
            
            for j, idx in enumerate(batch_indices):
                try:
                    _, label = train_ds[idx]
                    if isinstance(label, torch.Tensor):
                        label_tensor = label.float()
                    else:
                        label_tensor = torch.tensor(label, dtype=torch.float32)
                    
                    # Weight = max weight of all positive classes in this sample
                    # If no positive classes, use minimum weight
                    positive_mask = label_tensor > 0.5
                    if positive_mask.any():
                        sample_weights[idx] = class_weights[positive_mask].max().item()
                    else:
                        sample_weights[idx] = class_weights.min().item()
                except Exception:
                    # Use minimum weight for problematic samples
                    sample_weights[idx] = class_weights.min().item()
        
        # Normalize sample weights
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
        
        print(f"Sample weights: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")
        print(f"{'='*60}\n")
        
        # Create WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True  # Allow replacement to ensure rare classes appear frequently
        )
        
        self._train_sampler = sampler
        
        # Create dataloader with sampler
        self._train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        
        return self._train_dataloader
