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

# Loss functions
from reben_publication.losses import AsymmetricLoss

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
            class_weights: Optional[torch.Tensor] = None,
            threshold: float = 0.5,
            decoupling_epochs: Optional[int] = None,
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
        
        # Decoupling strategy parameters
        self.decoupling_epochs = decoupling_epochs
        self.current_phase = 1  # Phase 1: BCE + full training, Phase 2: ASL + frozen backbone
        if self.decoupling_epochs is not None:
            print(f"\n{'='*60}")
            print(f"DECOUPLING STRATEGY ENABLED")
            print(f"  Phase 1 (epochs 0-{decoupling_epochs-1}): BCE loss, full model training")
            print(f"  Phase 2 (epochs {decoupling_epochs}+): ASL loss, frozen backbone, head-only training")
            print(f"{'='*60}\n")
        
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
        
        # Loss functions: Use BCE for phase 1, ASL for phase 2 (if decoupling enabled)
        # Note: AsymmetricLoss doesn't support pos_weight, so class_weights are ignored
        # The loss handles class imbalance through asymmetric focusing instead
        if class_weights is not None:
            print(f"Note: class_weights provided but AsymmetricLoss doesn't use pos_weight.")
            print(f"  Class imbalance is handled through asymmetric focusing (gamma_neg=4, gamma_pos=1).")
        
        # Initialize both loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.asl_loss = AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0.05, eps=1e-8)
        
        # Set initial loss based on decoupling strategy
        if self.decoupling_epochs is not None:
            # Phase 1: Use BCE
            self.loss = self.bce_loss
            print(f"Initial loss: BCE (Phase 1 of decoupling strategy)")
        else:
            # Default: Use ASL
            self.loss = self.asl_loss
            print(f"Initial loss: ASL (standard training)")
        
        # Custom threshold for binary predictions (default 0.5)
        self.threshold = threshold
        if threshold != 0.5:
            print(f"Using custom threshold: {threshold} (default is 0.5)")
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

    def on_train_epoch_start(self):
        """Handle phase transition in decoupling strategy."""
        super().on_train_epoch_start()
        
        if self.decoupling_epochs is not None:
            current_epoch = self.current_epoch
            
            # Check if we need to transition from phase 1 to phase 2
            if current_epoch == self.decoupling_epochs and self.current_phase == 1:
                print(f"\n{'='*60}")
                print(f"TRANSITIONING TO PHASE 2: Freezing backbone, switching to ASL loss")
                print(f"{'='*60}\n")
                
                # Freeze backbone
                self._freeze_backbone()
                
                # Switch to ASL loss
                self.loss = self.asl_loss
                self.current_phase = 2
                
                # Reconfigure optimizer with only trainable parameters
                # Lightning will handle this automatically, but we need to update the optimizer
                if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self.trainer, 'optimizers'):
                    # Get current optimizer
                    optimizer = self.trainer.optimizers[0]
                    # Clear parameter groups and add only trainable parameters
                    optimizer.param_groups.clear()
                    
                    # Collect trainable parameters into a list
                    trainable_params = []
                    if hasattr(self.model, "backbone") and hasattr(self.model, "classifier"):
                        # DINOv3 case
                        trainable_params.extend(list(self.model.classifier.parameters()))
                        # Add patch embedding projection if present and trainable
                        try:
                            proj = None
                            if hasattr(self.model.backbone, 'embeddings') and hasattr(self.model.backbone.embeddings, 'patch_embeddings'):
                                pe = self.model.backbone.embeddings.patch_embeddings
                                if hasattr(pe, 'projection'):
                                    proj = pe.projection
                                elif isinstance(pe, torch.nn.Conv2d):
                                    proj = pe
                            if proj is not None:
                                trainable_params.extend([p for p in proj.parameters() if p.requires_grad])
                        except Exception:
                            pass
                    elif hasattr(self.model, "classifier"):
                        # ConfigILM case
                        trainable_params.extend(list(self.model.classifier.parameters()))
                    else:
                        # Fallback
                        trainable_params.extend([p for p in self.parameters() if p.requires_grad])
                    
                    # Add trainable parameters to optimizer
                    if trainable_params:
                        optimizer.add_param_group({"params": trainable_params, "lr": self.lr, "weight_decay": 0.01})
                    
                    num_trainable = sum(p.numel() for p in trainable_params)
                    print(f"Optimizer reconfigured for Phase 2: {num_trainable} trainable parameters")
    
    def _freeze_backbone(self):
        """Freeze the backbone parameters while keeping the head trainable."""
        if hasattr(self.model, 'backbone'):
            # DINOv3 case
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            # Keep classifier trainable
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            # Keep input adaptation layer trainable if it exists
            try:
                proj = None
                if hasattr(self.model.backbone, 'embeddings') and hasattr(self.model.backbone.embeddings, 'patch_embeddings'):
                    pe = self.model.backbone.embeddings.patch_embeddings
                    if hasattr(pe, 'projection'):
                        proj = pe.projection
                    elif isinstance(pe, torch.nn.Conv2d):
                        proj = pe
                if proj is not None:
                    for p in proj.parameters():
                        p.requires_grad = True
            except Exception:
                pass
            print("Backbone frozen (DINOv3)")
        else:
            # ConfigILM case - freeze all except classifier
            # ConfigILM models typically have a classifier attribute
            if hasattr(self.model, 'classifier'):
                # Freeze all parameters first
                for param in self.model.parameters():
                    param.requires_grad = False
                # Unfreeze classifier
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
                print("Backbone frozen (ConfigILM), classifier remains trainable")
            else:
                # Fallback: try to identify and freeze feature extractor
                # This is model-specific, so we'll do a best-effort approach
                for name, param in self.model.named_parameters():
                    if 'classifier' in name.lower() or 'head' in name.lower():
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                print("Backbone frozen (fallback method)")
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, y)
        self.log("train/loss", loss)
        # Log which phase we're in
        if self.decoupling_epochs is not None:
            self.log("train/phase", float(self.current_phase))
            self.log("train/loss_type", 0.0 if self.current_phase == 1 else 1.0)  # 0=BCE, 1=ASL
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            current_gpu_mem_mb = torch.cuda.memory_allocated(current_gpu) / 1024 ** 2
            self.log("train/GPU_memory_MB", current_gpu_mem_mb)
        return {"loss": loss}

    def configure_optimizers(self):
        # Check current phase based on epoch (if trainer is available) or use stored phase
        current_phase = self.current_phase
        if (self.decoupling_epochs is not None and 
            hasattr(self, 'trainer') and 
            self.trainer is not None and 
            hasattr(self.trainer, 'current_epoch')):
            # Determine phase based on current epoch
            if self.trainer.current_epoch < self.decoupling_epochs:
                current_phase = 1
            else:
                current_phase = 2
        
        # Handle decoupling strategy phase 2 (frozen backbone)
        if (self.decoupling_epochs is not None and current_phase == 2):
            # Phase 2: Only optimize classifier head
            param_groups = []
            
            if hasattr(self.model, "backbone") and hasattr(self.model, "classifier"):
                # DINOv3 case
                param_groups.append({"params": list(self.model.classifier.parameters())})
                # Add patch embedding projection if present and trainable
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
            elif hasattr(self.model, "classifier"):
                # ConfigILM case - only classifier
                param_groups.append({"params": list(self.model.classifier.parameters())})
            else:
                # Fallback: collect all trainable parameters
                trainable_params = [p for p in self.parameters() if p.requires_grad]
                param_groups.append({"params": trainable_params})
            
            optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=0.01)
        # If linear probing with DINOv3, optimize classifier head and the input adaptation layer
        elif hasattr(self, "model") and hasattr(self.model, "backbone") and hasattr(self.model, "classifier") and self.linear_probe:
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
            # Phase 1 or standard training: optimize all parameters
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

        logits = torch.cat([x["outputs"] for x in self.val_output_list])
        labels = torch.cat([x["labels"] for x in self.val_output_list]).long()

        # Apply custom threshold: convert logits to probabilities, then to binary predictions
        # Convert binary predictions back to logits for metrics that expect logits
        probs = torch.sigmoid(logits)
        binary_preds = (probs > self.threshold).float()
        # Convert binary predictions back to logits for metrics compatibility
        thresholded_logits = torch.where(
            binary_preds > 0.5,
            torch.tensor(10.0, device=logits.device),  # Large positive logit
            torch.tensor(-10.0, device=logits.device)  # Large negative logit
        )

        # Use thresholded logits for metrics that need binary predictions
        metrics_macro = self.val_metrics_macro(thresholded_logits, labels)
        self.log_dict(metrics_macro)
        self.val_metrics_macro.reset()

        metrics_micro = self.val_metrics_micro(thresholded_logits, labels)
        self.log_dict(metrics_micro)
        self.val_metrics_micro.reset()

        metrics_samples = self.val_metrics_samples(thresholded_logits.unsqueeze(-1), labels.unsqueeze(-1))
        metrics_samples = {k: v.mean() for k, v in metrics_samples.items()}
        self.log_dict(metrics_samples)
        self.val_metrics_samples.reset()

        # get class names from datamodule
        class_names = NEW_LABELS
        metrics_class = self.val_metrics_class(thresholded_logits, labels)
        classwise_acc = {
            f"val/ClasswiseAccuracy/{class_names[i]}": metrics_class["val/MultilabelAccuracy_class"][i]
            for i in range(len(class_names))
        }
        self.log_dict(classwise_acc)
        self.val_metrics_class.reset()
        
        # Log threshold used
        self.log("val/threshold", self.threshold)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.test_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.test_output_list]).mean()
        self.log("test/loss", avg_loss)

        logits = torch.cat([x["outputs"] for x in self.test_output_list])
        labels = torch.cat([x["labels"] for x in self.test_output_list]).long()

        # Apply custom threshold: convert logits to probabilities, then to binary predictions
        # Convert binary predictions back to logits for metrics that expect logits
        probs = torch.sigmoid(logits)
        binary_preds = (probs > self.threshold).float()
        # Convert binary predictions back to logits for metrics compatibility
        thresholded_logits = torch.where(
            binary_preds > 0.5,
            torch.tensor(10.0, device=logits.device),  # Large positive logit
            torch.tensor(-10.0, device=logits.device)  # Large negative logit
        )

        # Use thresholded logits for metrics that need binary predictions
        metrics_macro = self.test_metrics_macro(thresholded_logits, labels)
        self.log_dict(metrics_macro)
        self.test_metrics_macro.reset()

        metrics_micro = self.test_metrics_micro(thresholded_logits, labels)
        self.log_dict(metrics_micro)
        self.test_metrics_micro.reset()

        metrics_samples = self.test_metrics_samples(thresholded_logits.unsqueeze(-1), labels.unsqueeze(-1))
        metrics_samples = {k: v.mean() for k, v in metrics_samples.items()}
        self.log_dict(metrics_samples)
        self.test_metrics_samples.reset()

        class_names = NEW_LABELS
        metrics_class = self.test_metrics_class(thresholded_logits, labels)
        classwise_acc = {
            f"test/ClasswiseAccuracy/{class_names[i]}": metrics_class["test/MultilabelAccuracy_class"][i]
            for i in range(len(class_names))
        }
        self.log_dict(classwise_acc)
        self.test_metrics_class.reset()
        
        # Log threshold used
        self.log("test/threshold", self.threshold)

    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)
