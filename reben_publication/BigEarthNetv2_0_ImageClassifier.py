"""
This is a script for supervised image classification using the BigEarthNet v2.0 dataset.
"""
# import packages
from typing import List, Optional
import random

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
        
        # DEBUG: Print classifier configuration
        print(f"\n{'='*60}")
        print(f"DEBUG: BigEarthNetv2_0_ImageClassifier initialization")
        print(f"  linear_probe: {self.linear_probe}")
        print(f"  head_type: {self.head_type}")
        print(f"  mlp_hidden_dims: {self.mlp_hidden_dims}")
        print(f"  head_dropout: {self.head_dropout}")
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
        # Initialize loss without weights - will be updated in setup() if linear_probe=True
        self.loss = torch.nn.BCEWithLogitsLoss()
        # Initialize class_weights as None - will be set if linear_probe=True
        # This allows loading checkpoints that don't have this buffer
        self.register_buffer('class_weights', None)
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

    def _compute_class_weights(self):
        """
        Compute class weights from the training dataset for weighted loss.
        Returns a tensor of shape [num_classes] with pos_weight values.
        For multi-label classification: pos_weight[i] = num_negatives[i] / num_positives[i]
        """
        if not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
            print("WARNING: Cannot compute class weights - datamodule not available yet")
            return None
        
        try:
            train_ds = self.trainer.datamodule.train_ds
            num_classes = self.config.classes
            
            # Collect all labels from training dataset
            print("Computing class weights from training dataset...")
            all_labels = []
            
            # Sample a subset if dataset is very large (for efficiency)
            dataset_size = len(train_ds)
            sample_size = min(10000, dataset_size)  # Sample up to 10k examples
            
            indices = random.sample(range(dataset_size), sample_size) if sample_size < dataset_size else range(dataset_size)
            
            for idx in indices:
                try:
                    _, labels = train_ds[idx]
                    if isinstance(labels, torch.Tensor):
                        all_labels.append(labels)
                    else:
                        all_labels.append(torch.tensor(labels, dtype=torch.float32))
                except Exception:
                    # Skip problematic samples
                    continue
            
            if len(all_labels) == 0:
                print("WARNING: Could not collect any labels for weight computation")
                return None
            
            # Stack all labels
            labels_tensor = torch.stack(all_labels)  # Shape: [num_samples, num_classes]
            
            # Compute positive and negative counts for each class
            num_positives = labels_tensor.sum(dim=0)  # Shape: [num_classes]
            num_negatives = labels_tensor.shape[0] - num_positives  # Shape: [num_classes]
            
            # Compute pos_weight: weight for positive examples
            # pos_weight[i] = num_negatives[i] / num_positives[i]
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            pos_weight = num_negatives / (num_positives + epsilon)
            
            # Clip extreme weights to reasonable range (e.g., 0.1 to 100)
            pos_weight = torch.clamp(pos_weight, min=0.1, max=100.0)
            
            print(f"Computed class weights (pos_weight):")
            for i, weight in enumerate(pos_weight):
                class_name = NEW_LABELS[i] if i < len(NEW_LABELS) else f"Class_{i}"
                print(f"  {class_name}: {weight:.4f} (pos: {num_positives[i].item()}, neg: {num_negatives[i].item()})")
            
            return pos_weight
            
        except Exception as e:
            print(f"WARNING: Error computing class weights: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Override to handle missing keys for class_weights and loss.pos_weight
        when loading checkpoints from models trained without weighted loss.
        Also handles unexpected keys when loading checkpoints with weighted loss into
        a model that hasn't initialized weighted loss yet.
        """
        # Keys that are allowed to be missing (from old checkpoints)
        allowed_missing_keys = [
            f"{prefix}class_weights",
            f"{prefix}loss.pos_weight",
        ]
        
        # Keys that are allowed to be unexpected (from checkpoints with weighted loss)
        # These might exist in checkpoints but not in current model if setup() hasn't run yet
        allowed_unexpected_keys = [
            f"{prefix}loss.pos_weight",
        ]
        
        # Check which allowed keys are actually missing
        missing_allowed = [k for k in allowed_missing_keys if k in missing_keys]
        
        # Check which allowed keys are unexpected
        unexpected_allowed = [k for k in allowed_unexpected_keys if k in unexpected_keys]
        
        # Remove allowed missing keys from the list (modify in place)
        for key in allowed_missing_keys:
            if key in missing_keys:
                missing_keys.remove(key)
        
        # Remove allowed unexpected keys from the list (modify in place)
        for key in allowed_unexpected_keys:
            if key in unexpected_keys:
                unexpected_keys.remove(key)
                # Also remove from state_dict so it doesn't cause issues
                if key in state_dict:
                    del state_dict[key]
        
        # Call parent with filtered missing and unexpected keys
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, 
            missing_keys, unexpected_keys, error_msgs
        )
        
        # Print informative messages about missing allowed keys
        if f"{prefix}class_weights" in missing_allowed:
            print(f"Note: Checkpoint does not contain 'class_weights' buffer. "
                  f"This is normal for checkpoints trained without weighted loss.")
        if f"{prefix}loss.pos_weight" in missing_allowed:
            print(f"Note: Checkpoint does not contain 'loss.pos_weight' parameter. "
                  f"This is normal for checkpoints trained without weighted loss.")
        if f"{prefix}loss.pos_weight" in unexpected_allowed:
            print(f"Note: Checkpoint contains 'loss.pos_weight' but current model doesn't have it yet. "
                  f"This is normal - weights will be recomputed in setup() if linear_probe=True.")

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle missing keys for class_weights and loss.pos_weight.
        This allows loading checkpoints that don't have weighted loss.
        """
        # Keys that are allowed to be missing (from old checkpoints)
        allowed_missing_keys = [
            "class_weights",
            "loss.pos_weight",
        ]
        
        # Get current model's state dict to see what keys we expect
        model_keys = set(self.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        # Find missing keys
        missing_keys = model_keys - checkpoint_keys
        missing_allowed = [k for k in missing_keys if k in allowed_missing_keys]
        
        # Remove allowed missing keys from consideration
        actual_missing = missing_keys - set(allowed_missing_keys)
        
        # Remove allowed unexpected keys from state_dict
        for key in allowed_missing_keys:
            if key in state_dict:
                del state_dict[key]
        
        # If strict=True and there are non-allowed missing keys, raise error
        if strict and actual_missing:
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n"
                f"        Missing key(s) in state_dict: {sorted(actual_missing)}."
            )
        
        # Call parent with modified state_dict
        # Use strict=False if we have allowed missing keys, otherwise use the original strict value
        effective_strict = strict and not missing_allowed
        super().load_state_dict(state_dict, strict=effective_strict)
        
        # Print informative messages about missing allowed keys
        if "class_weights" in missing_allowed:
            print(f"Note: Checkpoint does not contain 'class_weights' buffer. "
                  f"This is normal for checkpoints trained without weighted loss.")
        if "loss.pos_weight" in missing_allowed:
            print(f"Note: Checkpoint does not contain 'loss.pos_weight' parameter. "
                  f"This is normal for checkpoints trained without weighted loss.")

    def setup(self, stage: str):
        """
        Lightning hook called after datamodule setup.
        Compute class weights and update loss function if linear_probe=True.
        """
        super().setup(stage)
        
        if self.linear_probe and stage == "fit":
            print("\n" + "="*60)
            print("Linear probe mode detected - computing class weights for weighted loss")
            print("="*60)
            
            pos_weight = self._compute_class_weights()
            if pos_weight is not None:
                # Update the buffer (it was initialized as None)
                self.class_weights = pos_weight
                # Update loss function with computed weights
                # Use the registered buffer which will be on the correct device
                self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
                print(f"Updated loss function with class weights")
                print("="*60 + "\n")
            else:
                print("WARNING: Could not compute class weights, using unweighted loss")
                print("="*60 + "\n")

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
