"""
Training script for multi-backbone model with concatenated features and linear classifier.
Supports training multiple backbones with different input channels and easily training just the classifier.
"""
import sys
from pathlib import Path

# Add parent directory to path to allow importing reben_publication
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import typer
from typing import List, Optional
from configilm.extra.BENv2_utils import resolve_data_dir, NEW_LABELS
from configilm.extra.CustomTorchClasses import LinearWarmupCosineAnnealingLR
from configilm.metrics import get_classification_metric_collection
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule

from models.multimodal import MultimodalLateFusionClassifier
from models.backbone_config import BackboneConfig
from scripts.utils import get_benv2_dir_dict, default_trainer, default_dm

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"



class MultiBackboneLightningModule(pl.LightningModule):
    """
    Lightning module wrapper for MultimodalLateFusionClassifier.
    """
    
    def __init__(
        self,
        model: MultimodalLateFusionClassifier,
        lr: float = 1e-3,
        warmup: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.warmup = None if warmup is None or warmup < 0 else warmup
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        num_classes = model.num_classes
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
        self.val_output_list = []
        self.test_output_list = []
    
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
        # Check if backbones are frozen
        backbones_frozen = getattr(self.model, 'freeze_backbones', False)
        
        # If backbones are frozen, only optimize classifier and fusion
        if backbones_frozen:
            # Get parameters from classifier and fusion
            params = list(self.model.classifier.parameters())
            if hasattr(self.model, 'fusion') and hasattr(self.model.fusion, 'parameters'):
                params.extend(list(self.model.fusion.parameters()))
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        
        max_intervals = int(
            self.trainer.max_epochs * len(self.trainer.datamodule.train_ds) / self.trainer.datamodule.batch_size
        )
        if self.warmup is not None:
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


def main(
        # Backbone configurations (format: "name:architecture:channels")
        # Example: "resnet18:resnet18:3" "resnet50:resnet50:10"
        backbones: List[str] = typer.Option(
            ["resnet18:resnet18:3", "resnet18:resnet18:10"],
            help='List of backbones in format "name:architecture:channels". Example: "backbone1:resnet18:3 backbone2:resnet50:10"'
        ),
        image_size: int = typer.Option(120, help="Input image size"),
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(32, help="Batch size"),
        drop_rate: float = typer.Option(0.15, help="Dropout rate"),
        drop_path_rate: float = typer.Option(0.0, help="Drop path rate"),
        warmup: int = typer.Option(1000, help="Warmup steps, set to -1 for automatic calculation"),
        workers: int = typer.Option(8, help="Number of workers"),
        freeze_backbones: bool = typer.Option(False, help="Freeze backbone parameters and only train classifier"),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        test_run: bool = typer.Option(True, help="Run training with fewer epochs and batches"),
        resume_from: str = typer.Option(None, help="Path to checkpoint file to resume training from. "
                                                   "Can be a full path or 'best'/'last'."),
):
    """
    Train a multi-backbone model with concatenated features and a linear classifier.
    
    Examples:
        # Train with two ResNet18 backbones (3 channels + 10 channels)
        python train_multi_backbone.py --backbones "resnet18_3ch:resnet18:3" "resnet18_10ch:resnet18:10" --epochs 50
        
        # Train only the classifier (freeze backbones)
        python train_multi_backbone.py --backbones "resnet18_3ch:resnet18:3" "resnet50_10ch:resnet50:10" --freeze-backbones --lr 0.01
        
        # Train with custom hyperparameters
        python train_multi_backbone.py --backbones "backbone1:resnet18:3" "backbone2:resnet34:10" --epochs 200 --lr 0.0005 --bs 128
    """
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    
    # FIXED MODEL PARAMETERS
    num_classes = 19
    
    # Parse backbone configurations
    backbone_configs = []
    total_channels = 0
    for backbone_str in backbones:
        parts = backbone_str.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid backbone format: {backbone_str}. Expected 'name:architecture:channels'")
        name, architecture, channels_str = parts
        try:
            channels = int(channels_str)
        except ValueError:
            raise ValueError(f"Invalid channels value: {channels_str}. Must be an integer.")
        
        # Check if DINOv3
        dinov3_model_name = None
        if architecture.startswith('dinov3'):
            # Auto-determine DINOv3 model name
            if 'small' in architecture.lower() or 's' in architecture.lower():
                dinov3_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
            elif 'base' in architecture.lower() or 'b' in architecture.lower():
                dinov3_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            elif 'large' in architecture.lower() or 'l' in architecture.lower():
                dinov3_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            elif 'giant' in architecture.lower() or 'g' in architecture.lower():
                dinov3_model_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
            else:
                dinov3_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        
        config = BackboneConfig(
            name=name,
            input_channels=channels,
            architecture=architecture,
            pretrained=True,
            dinov3_model_name=dinov3_model_name,
        )
        backbone_configs.append(config)
        total_channels += channels
        print(f"Added backbone: {name} ({architecture}) with {channels} input channels")
    
    print(f"Total input channels: {total_channels}")
    
    # set seed
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    warmup = None if warmup == -1 else warmup
    assert warmup is None or warmup > 0, "Warmup steps must be positive or -1 for automatic calculation"
    
    # Create model
    model_base = MultimodalLateFusionClassifier(
        backbone_configs=backbone_configs,
        image_size=image_size,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        freeze_backbones=freeze_backbones,
    )
    
    model = MultiBackboneLightningModule(model_base, lr=lr, warmup=warmup)
    
    hparams = {
        "backbones": [f"{c.name}:{c.architecture}:{c.input_channels}" for c in backbone_configs],
        "freeze_backbones": freeze_backbones,
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "batch_size": bs,
        "workers": workers,
        "channels": total_channels,
        "dropout": drop_rate,
        "drop_path_rate": drop_path_rate,
        "warmup": warmup,
        "image_size": image_size,
    }
    
    trainer = default_trainer(hparams, use_wandb, test_run)
    
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
    
    # Create data module with total number of channels
    hparams_for_dm = {
        "batch_size": bs,
        "workers": workers,
        "channels": total_channels,
    }
    dm = default_dm(hparams_for_dm, data_dirs, image_size)
    
    # Handle checkpoint resume
    ckpt_path = None
    if resume_from is not None:
        if resume_from.lower() in ["best", "last"]:
            ckpt_path = resume_from.lower()
            print(f"Resuming from {resume_from} checkpoint (will be resolved by Lightning)")
        else:
            ckpt_path_obj = Path(resume_from)
            if not ckpt_path_obj.exists():
                ckpt_path_obj = Path("./checkpoints") / resume_from
                if not ckpt_path_obj.exists():
                    raise FileNotFoundError(
                        f"Checkpoint not found: {resume_from}\n"
                        f"Tried: {resume_from} and ./checkpoints/{resume_from}"
                    )
            ckpt_path = str(ckpt_path_obj.resolve())
            print(f"Resuming training from checkpoint: {ckpt_path}")
    
    trainer.fit(model, dm, ckpt_path=ckpt_path)
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    
    print("=== Training finished ===")
    print(f"Test results: {results[0]}")


if __name__ == "__main__":
    typer.run(main)

