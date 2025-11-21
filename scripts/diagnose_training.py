"""
Training Diagnostic Script

This script performs comprehensive diagnostics on training runs:
1. Train vs validation performance comparison
2. Per-class confusion matrix and metrics
3. Training curves (loss, accuracy, F1 per epoch)
4. Data pipeline checks (RGB channels, normalization, augmentations)
5. Model configuration analysis

Usage:
    python diagnose_training.py --checkpoint <path_to_checkpoint.ckpt> [options]
    
The script will automatically discover related logs and directories from the checkpoint path.
It aligns with the train_multimodal.py pipeline structure.
"""
import sys
from pathlib import Path
import argparse
import warnings
import json
import yaml
from datetime import datetime
warnings.filterwarnings('ignore')

# Add parent directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

from configilm.extra.BENv2_utils import NEW_LABELS, resolve_data_dir, STANDARD_BANDS
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
from configilm.ConfigILM import ILMConfiguration, ILMType
from multimodal.lightning_module import MultiModalLightningModule
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from scripts.utils import (
    default_dm,
    get_benv2_dir_dict,
    resolve_checkpoint_path,
    get_bands,
    DEFAULT_CKPT_LOG_ROOT,
)

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrainingDiagnostics:
    """Comprehensive training diagnostics aligned with train_multimodal.py pipeline."""
    
    def __init__(self, checkpoint_path: Path, config_path: Optional[Path] = None):
        """
        Initialize diagnostics from checkpoint path.
        
        Args:
            checkpoint_path: Path to checkpoint file (.ckpt)
            config_path: Optional path to config YAML file for data directories
        """
        # Resolve checkpoint path using same logic as train_multimodal.py
        try:
            self.checkpoint_path = resolve_checkpoint_path(str(checkpoint_path))
        except FileNotFoundError:
            # Fallback to direct path resolution
            self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.config_path = config_path
        self.checkpoint_dir = self.checkpoint_path.parent
        
        # Discover run directory (checkpoint might be in ckpt_logs/<run_name>/checkpoints/)
        self.run_dir = None
        if DEFAULT_CKPT_LOG_ROOT.exists():
            # Check if checkpoint is inside a run directory
            try:
                relative_path = self.checkpoint_path.relative_to(DEFAULT_CKPT_LOG_ROOT)
                # Pattern: <run_name>/checkpoints/<checkpoint_file>
                parts = relative_path.parts
                if len(parts) >= 2 and parts[1] == "checkpoints":
                    self.run_dir = DEFAULT_CKPT_LOG_ROOT / parts[0]
            except ValueError:
                pass
        
        # Auto-discover lightning logs directory
        self.lightning_logs_dir = None
        possible_log_dirs = [
            self.run_dir / "lightning_logs" if self.run_dir else None,
            self.checkpoint_dir.parent / "lightning_logs",
            self.checkpoint_dir / "lightning_logs",
            self.checkpoint_dir.parent.parent / "lightning_logs",
            Path("lightning_logs"),
        ]
        
        for log_dir in possible_log_dirs:
            if log_dir and log_dir.exists():
                # Find version directories
                versions = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("version_")])
                if versions:
                    csv_path = versions[-1] / "metrics.csv"
                    if csv_path.exists():
                        self.lightning_logs_dir = versions[-1]
                        break
        
        self.num_classes = len(NEW_LABELS)
        self.class_names = NEW_LABELS
        
        # Storage for metrics and config
        self.train_metrics = {}
        self.val_metrics = {}
        self.epochs = []
        self.hparams = {}
        self.model_config = {}
        
        # Detect model type from checkpoint
        self.model_type = self._detect_model_type()
        self._load_checkpoint_info()
        
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Model type: {self.model_type}")
        if self.run_dir:
            print(f"Run directory: {self.run_dir}")
        if self.lightning_logs_dir:
            print(f"Found Lightning logs: {self.lightning_logs_dir}")
        else:
            print("Warning: Lightning logs not found. Training curves may be unavailable.")
    
    def _load_checkpoint_info(self):
        """Load hyperparameters and model configuration from checkpoint."""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            if 'hyper_parameters' in checkpoint:
                self.hparams = checkpoint['hyper_parameters'].copy()
            
            # Extract model-specific config
            if self.model_type == 'multimodal':
                self.model_config = {
                    'fusion_type': self.hparams.get('fusion_type', 'concat'),
                    'classifier_type': self.hparams.get('classifier_type', 'linear'),
                    'use_s1': self.hparams.get('use_s1', False),
                    'use_resnet': self.hparams.get('use_resnet_flag', True),
                    'resnet_enabled': self.hparams.get('resnet_enabled', True),
                    'resnet_input_channels': self.hparams.get('resnet_input_channels', 11),
                    'dinov3_model_name': self.hparams.get('dinov3_model_name', None),
                    'bandconfig': self.hparams.get('bandconfig', 's2s1'),
                }
        except Exception as e:
            print(f"Warning: Could not load checkpoint info: {e}")
    
    def _detect_model_type(self) -> str:
        """Detect model type from checkpoint."""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Check hyperparameters for architecture hint
            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                architecture = hparams.get('architecture', '')
                
                # Check if it's multimodal (has multimodal-specific keys)
                if 'fusion_type' in hparams or 'use_resnet_flag' in hparams:
                    return 'multimodal'
                
                # Check if it's BigEarthNetv2_0 (standard architecture)
                if architecture and not architecture.startswith('multimodal'):
                    return 'bigearthnet'
            
            # Check state_dict keys for model structure
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                keys = list(state_dict.keys())
                
                # Multimodal has dinov3_backbone and resnet_backbone
                if any('dinov3_backbone' in k for k in keys):
                    if any('resnet_backbone' in k for k in keys):
                        return 'multimodal'
                    # Could be RGB-only multimodal (no resnet)
                    if 'fusion' in str(keys[0]) or 'classifier' in str(keys[0]):
                        return 'multimodal'
                
                # BigEarthNetv2_0 has model.backbone or model.model
                if any('model.backbone' in k or 'model.model' in k for k in keys):
                    return 'bigearthnet'
            
            # Default: try to infer from checkpoint path/name
            checkpoint_name = self.checkpoint_path.name.lower()
            if 'multimodal' in checkpoint_name:
                return 'multimodal'
            
            # Default to bigearthnet (more common)
            return 'bigearthnet'
            
        except Exception as e:
            print(f"Warning: Could not detect model type: {e}")
            return 'bigearthnet'  # Default fallback
    
    def _compute_metrics_manual(self, y_true, y_pred):
        """Manually compute precision, recall, F1, and support."""
        num_classes = y_true.shape[1]
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        support = np.zeros(num_classes, dtype=int)
        
        for i in range(num_classes):
            tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
            
            support[i] = int(np.sum(y_true[:, i]))
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0
        
        return precision, recall, f1, support
    
    def load_metrics_from_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load metrics from Lightning CSV logger."""
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV log file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        return df
    
    def find_lightning_logs(self) -> Optional[Path]:
        """Find Lightning log CSV file."""
        if self.lightning_logs_dir:
            csv_path = self.lightning_logs_dir / "metrics.csv"
            if csv_path.exists():
                return csv_path
        
        # Fallback: search common locations
        search_dirs = [
            self.run_dir / "lightning_logs" if self.run_dir else None,
            self.checkpoint_dir / "lightning_logs",
            self.checkpoint_dir.parent / "lightning_logs",
            Path("lightning_logs"),
            Path(".") / "lightning_logs",
        ]
        
        for log_dir in search_dirs:
            if log_dir and log_dir.exists():
                # Find version directories
                versions = sorted([d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("version_")])
                if versions:
                    csv_path = versions[-1] / "metrics.csv"
                    if csv_path.exists():
                        return csv_path
        
        return None
    
    def load_training_history(self) -> Dict:
        """Load training history from Lightning CSV logs."""
        metrics_df = None
        
        # Try to load from discovered logs directory
        if self.lightning_logs_dir:
            csv_path = self.lightning_logs_dir / "metrics.csv"
            if csv_path.exists():
                metrics_df = self.load_metrics_from_csv(csv_path)
        
        # Fallback: search for logs
        if metrics_df is None:
            csv_path = self.find_lightning_logs()
            if csv_path:
                metrics_df = self.load_metrics_from_csv(csv_path)
        
        if metrics_df is None:
            print("Warning: Could not find Lightning CSV logs. Training curves will be unavailable.")
            return {}
        
        # Extract metrics
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_f1_macro': [],
            'val_f1_macro': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_map': [],
            'val_map': [],
        }
        
        # Extract epoch numbers
        if 'epoch' in metrics_df.columns:
            history['epoch'] = metrics_df['epoch'].dropna().unique().tolist()
        else:
            # Infer from step
            if 'step' in metrics_df.columns:
                # Assume validation happens every N steps
                history['epoch'] = list(range(len(metrics_df[metrics_df['val/loss'].notna()])))
        
        # Extract metrics - map Lightning metric names to our history keys
        metric_mapping = {
            'train/loss': 'train_loss',
            'val/loss': 'val_loss',
            'train/MultilabelF1Score_macro': 'train_f1_macro',
            'val/MultilabelF1Score_macro': 'val_f1_macro',
            'train/MultilabelAccuracy': 'train_accuracy',
            'val/MultilabelAccuracy': 'val_accuracy',
            'train/MultilabelAveragePrecision_macro': 'train_map',
            'val/MultilabelAveragePrecision_macro': 'val_map',
        }
        
        for metric_name, history_key in metric_mapping.items():
            if metric_name in metrics_df.columns:
                values = metrics_df[metric_name].dropna().tolist()
                history[history_key] = values
        
        # Align lengths
        list_lengths = [len(v) for v in history.values() if isinstance(v, list) and len(v) > 0]
        if list_lengths:
            max_len = max(list_lengths)
            for key in history:
                if isinstance(history[key], list) and len(history[key]) < max_len:
                    history[key] = history[key] + [None] * (max_len - len(history[key]))
        
        return history
    
    def check_train_vs_val_performance(self, history: Dict) -> Dict:
        """Check train vs validation performance."""
        print("\n" + "="*80)
        print("1. TRAIN vs VALIDATION PERFORMANCE")
        print("="*80)
        
        results = {}
        
        if not history or 'val_loss' not in history or not history['val_loss']:
            print("Warning: No validation metrics found in history.")
            return results
        
        # Get final metrics (filter out None values)
        def get_last_valid(lst):
            if not lst:
                return None
            valid = [v for v in lst if v is not None]
            return valid[-1] if valid else None
        
        train_loss_final = get_last_valid(history.get('train_loss', []))
        val_loss_final = get_last_valid(history.get('val_loss', []))
        train_f1_final = get_last_valid(history.get('train_f1_macro', []))
        val_f1_final = get_last_valid(history.get('val_f1_macro', []))
        train_map_final = get_last_valid(history.get('train_map', []))
        val_map_final = get_last_valid(history.get('val_map', []))
        
        results['train_loss'] = train_loss_final
        results['val_loss'] = val_loss_final
        results['train_f1_macro'] = train_f1_final
        results['val_f1_macro'] = val_f1_final
        results['train_map'] = train_map_final
        results['val_map'] = val_map_final
        
        print(f"\nFinal Metrics:")
        if train_loss_final and val_loss_final:
            print(f"  Train Loss:   {train_loss_final:.4f}")
            print(f"  Val Loss:      {val_loss_final:.4f}")
            gap_loss = train_loss_final - val_loss_final
            results['loss_gap'] = gap_loss
            print(f"  Loss Gap:      {gap_loss:.4f} {'(train > val: possible underfitting)' if gap_loss > 0.1 else '(train < val: possible overfitting)' if gap_loss < -0.1 else '(good)'}")
        
        if train_f1_final and val_f1_final:
            print(f"  Train F1 Macro: {train_f1_final:.4f}")
            print(f"  Val F1 Macro:   {val_f1_final:.4f}")
            gap_f1 = train_f1_final - val_f1_final
            results['f1_gap'] = gap_f1
            print(f"  F1 Gap:         {gap_f1:.4f} {'(train >> val: OVERFITTING!)' if gap_f1 > 0.1 else '(train ≈ val ≈ low: underfitting/optimization issue)' if abs(gap_f1) < 0.05 and val_f1_final < 0.5 else '(good)'}")
        
        if train_map_final and val_map_final:
            print(f"  Train mAP:      {train_map_final:.4f}")
            print(f"  Val mAP:        {val_map_final:.4f}")
            gap_map = train_map_final - val_map_final
            results['map_gap'] = gap_map
            print(f"  mAP Gap:        {gap_map:.4f}")
        
        # Diagnose issues
        diagnosis = []
        print(f"\nDiagnosis:")
        if train_loss_final and val_loss_final:
            if train_loss_final < val_loss_final - 0.1:
                msg = "⚠️  OVERFITTING DETECTED: Train loss << Val loss"
                print(f"  {msg}")
                print("     → Solutions: regularization, more data/augmentation, early stopping")
                diagnosis.append(msg)
            elif train_loss_final > val_loss_final + 0.1:
                msg = "⚠️  UNDERFITTING DETECTED: Train loss >> Val loss"
                print(f"  {msg}")
                print("     → Solutions: increase model capacity, train longer, check LR")
                diagnosis.append(msg)
            else:
                msg = "✓ Train/Val loss gap is reasonable"
                print(f"  {msg}")
                diagnosis.append(msg)
        
        if train_f1_final and val_f1_final:
            if train_f1_final > val_f1_final + 0.1:
                msg = "⚠️  OVERFITTING: Train F1 >> Val F1"
                print(f"  {msg}")
                diagnosis.append(msg)
            elif train_f1_final < 0.5 and val_f1_final < 0.5:
                msg = "⚠️  UNDERFITTING or OPTIMIZATION ISSUE: Both train and val F1 are low"
                print(f"  {msg}")
                print("     → Check: learning rate, training time, head/backbone setup")
                diagnosis.append(msg)
            elif train_f1_final > 0.7 and val_f1_final < 0.5:
                msg = "⚠️  CLASS IMBALANCE or PER-CLASS FAILURE: Train high but F1 macro low"
                print(f"  {msg}")
                print("     → Check per-class metrics (see section 2)")
                diagnosis.append(msg)
        
        results['diagnosis'] = diagnosis
        return results
    
    def _reconstruct_multimodal_config(self, hparams: dict) -> dict:
        """Reconstruct multimodal model configuration from checkpoint hyperparameters."""
        config = {
            "backbones": {
                "dinov3": {
                    "model_name": hparams.get('dinov3_model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m'),
                    "pretrained": hparams.get('dinov3_pretrained', True),
                    "freeze": hparams.get('dinov3_freeze', False),
                    "lr": hparams.get('dinov3_lr', 1e-4),
                },
                "resnet101": {
                    "input_channels": hparams.get('resnet_input_channels', 11),
                    "pretrained": hparams.get('resnet_pretrained', True),
                    "freeze": hparams.get('resnet_freeze', False),
                    "lr": hparams.get('resnet_lr', 1e-4),
                    "enabled": hparams.get('resnet_enabled', True),
                },
            },
            "fusion": {
                "type": hparams.get('fusion_type', 'concat'),
            },
            "classifier": {
                "type": hparams.get('classifier_type', 'linear'),
                "num_classes": 19,
                "drop_rate": hparams.get('dropout', hparams.get('drop_rate', 0.15)),
            },
            "image_size": 120,
        }
        
        if hparams.get('fusion_output_dim'):
            config["fusion"]["output_dim"] = hparams['fusion_output_dim']
        
        if hparams.get('classifier_type') == 'mlp':
            config["classifier"]["hidden_dim"] = hparams.get('classifier_hidden_dim', 512)
        
        return config
    
    def _reconstruct_ilm_config(self, hparams: dict) -> ILMConfiguration:
        """Reconstruct ILMConfiguration from checkpoint hyperparameters."""
        # Extract parameters with defaults
        architecture = hparams.get('architecture', None)
        
        # If architecture not in hparams, try to infer from checkpoint filename
        if architecture is None:
            checkpoint_name = self.checkpoint_path.name.lower()
            # Check for common architecture patterns
            if 'dinov3-base' in checkpoint_name or 'dinov3-b' in checkpoint_name:
                architecture = 'dinov3-base'
            elif 'dinov3-large' in checkpoint_name or 'dinov3-l' in checkpoint_name:
                architecture = 'dinov3-large'
            elif 'dinov3-small' in checkpoint_name or 'dinov3-s' in checkpoint_name:
                architecture = 'dinov3-small'
            elif 'dinov3-giant' in checkpoint_name or 'dinov3-g' in checkpoint_name:
                architecture = 'dinov3-giant'
            elif 'resnet' in checkpoint_name:
                # Try to extract resnet version
                if 'resnet101' in checkpoint_name:
                    architecture = 'resnet101'
                elif 'resnet50' in checkpoint_name:
                    architecture = 'resnet50'
                elif 'resnet18' in checkpoint_name:
                    architecture = 'resnet18'
                else:
                    architecture = 'resnet101'  # Default
            else:
                architecture = 'resnet101'  # Default fallback
        
        channels = hparams.get('channels', None)
        # If channels not in hparams, try to infer from filename or use default
        if channels is None:
            # Check filename for channel hints (e.g., "3" in "dinov3-base-42-3-...")
            checkpoint_name = self.checkpoint_path.name
            parts = checkpoint_name.split('-')
            for i, part in enumerate(parts):
                if part.isdigit() and i > 0:  # Not the first part
                    try:
                        channels = int(part)
                        if 1 <= channels <= 20:  # Reasonable range
                            break
                    except ValueError:
                        pass
            if channels is None:
                channels = 12  # Default fallback
        
        drop_rate = hparams.get('dropout', hparams.get('drop_rate', 0.15))
        drop_path_rate = hparams.get('drop_path_rate', 0.15)
        num_classes = 19
        img_size = 120
        
        # Create ILMConfiguration
        config = ILMConfiguration(
            network_type=ILMType.IMAGE_CLASSIFICATION,
            classes=num_classes,
            image_size=img_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            timm_model_name=architecture,
            channels=channels,
        )
        return config
    
    def _load_model(self):
        """Load model from checkpoint based on detected type."""
        if self.model_type == 'multimodal':
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            
            # Reconstruct config
            config = self._reconstruct_multimodal_config(hparams)
            
            # Extract other parameters
            lr = hparams.get('lr', 0.001)
            warmup = hparams.get('warmup', None)
            dinov3_checkpoint = hparams.get('dinov3_checkpoint', None)
            resnet_checkpoint = hparams.get('resnet_checkpoint', None)
            freeze_dinov3 = hparams.get('dinov3_freeze', False)
            freeze_resnet = hparams.get('resnet_freeze', False)
            dinov3_model_name = hparams.get('dinov3_model_name', config['backbones']['dinov3']['model_name'])
            
            # Resolve checkpoint paths if provided
            if dinov3_checkpoint:
                try:
                    dinov3_checkpoint = str(resolve_checkpoint_path(dinov3_checkpoint))
                except FileNotFoundError:
                    dinov3_checkpoint = None
            
            if resnet_checkpoint:
                try:
                    resnet_checkpoint = str(resolve_checkpoint_path(resnet_checkpoint))
                except FileNotFoundError:
                    resnet_checkpoint = None
            
            return MultiModalLightningModule.load_from_checkpoint(
                str(self.checkpoint_path),
                config=config,
                lr=lr,
                warmup=warmup,
                dinov3_checkpoint=dinov3_checkpoint,
                resnet_checkpoint=resnet_checkpoint,
                freeze_dinov3=freeze_dinov3,
                freeze_resnet=freeze_resnet,
                dinov3_model_name=dinov3_model_name,
            )
        else:  # bigearthnet
            # Need to reconstruct ILMConfiguration from checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            
            # Reconstruct config
            config = self._reconstruct_ilm_config(hparams)
            
            # Extract other parameters
            lr = hparams.get('lr', 0.001)
            warmup = hparams.get('warmup', None)
            dinov3_model_name = hparams.get('dinov3_model_name', None)
            linear_probe = hparams.get('linear_probe', False)
            head_type = hparams.get('head_type', 'linear')
            mlp_hidden_dims = hparams.get('head_mlp_dims', None)
            head_dropout = hparams.get('head_dropout', None)
            
            # Load model with reconstructed config
            return BigEarthNetv2_0_ImageClassifier.load_from_checkpoint(
                str(self.checkpoint_path),
                config=config,
                lr=lr,
                warmup=warmup,
                dinov3_model_name=dinov3_model_name,
                linear_probe=linear_probe,
                head_type=head_type,
                mlp_hidden_dims=mlp_hidden_dims,
                head_dropout=head_dropout,
            )
    
    def _forward_pass(self, model, x):
        """Perform forward pass based on model type."""
        if self.model_type == 'multimodal':
            rgb_data, non_rgb_data = model._split_modalities(x)
            return model(rgb_data, non_rgb_data)
        else:  # bigearthnet
            # BigEarthNetv2_0_ImageClassifier wraps the model, so we call model.model(x)
            # which is the underlying ConfigILM model
            return model.model(x)
    
    def compute_confusion_matrix(self, data_module=None, num_samples: int = 1000):
        """Compute confusion matrix from checkpoint."""
        print("\n" + "="*80)
        print("2. PER-CLASS CONFUSION MATRIX")
        print("="*80)
        print(f"Loading checkpoint: {self.checkpoint_path}")
        print(f"Model type: {self.model_type}")
        
        try:
            # Load model
            model = self._load_model()
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Get data loader
            if data_module is None:
                print("Warning: No data module provided. Cannot compute confusion matrix.")
                return None
            
            val_loader = data_module.val_dataloader()
            
            # Collect predictions and labels
            all_preds = []
            all_labels = []
            
            print(f"Computing predictions on {num_samples} samples...")
            with torch.no_grad():
                for i, (x, y) in enumerate(val_loader):
                    if i * val_loader.batch_size >= num_samples:
                        break
                    
                    x = x.to(device)
                    y = y.to(device)
                    
                    # Use appropriate forward pass based on model type
                    logits = self._forward_pass(model, x)
                    preds = torch.sigmoid(logits) > 0.5
                    
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(y.cpu().numpy())
            
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            
            # Compute per-class metrics
            try:
                from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
            except ImportError:
                print("Warning: sklearn not available. Computing metrics manually...")
                # Manual computation
                precision, recall, f1, support = self._compute_metrics_manual(all_labels, all_preds)
            else:
                precision, recall, f1, support = precision_recall_fscore_support(
                    all_labels, all_preds, average=None, zero_division=0
                )
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Class': self.class_names,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Support': support
            })
            
            print("\nPer-Class Metrics:")
            print(results_df.to_string(index=False))
            
            # Identify problematic classes
            print("\nProblematic Classes (F1 < 0.3 or Support < 100):")
            problematic = results_df[(results_df['F1'] < 0.3) | (results_df['Support'] < 100)]
            if len(problematic) > 0:
                print(problematic.to_string(index=False))
            else:
                print("  None found!")
            
            return results_df
            
        except Exception as e:
            print(f"Error computing confusion matrix: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_training_curves(self, history: Dict, output_dir: Path):
        """Plot training curves."""
        print("\n" + "="*80)
        print("3. TRAINING CURVES")
        print("="*80)
        
        if not history:
            print("Warning: No training history available. Skipping plots.")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss
        ax = axes[0, 0]
        if history.get('train_loss') and history.get('val_loss'):
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: F1 Macro
        ax = axes[0, 1]
        if history.get('train_f1_macro') and history.get('val_f1_macro'):
            epochs = range(1, len(history['train_f1_macro']) + 1)
            ax.plot(epochs, history['train_f1_macro'], 'b-', label='Train F1 Macro', linewidth=2)
            ax.plot(epochs, history['val_f1_macro'], 'r-', label='Val F1 Macro', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Macro')
            ax.set_title('F1 Macro Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Accuracy
        ax = axes[1, 0]
        if history.get('train_accuracy') and history.get('val_accuracy'):
            epochs = range(1, len(history['train_accuracy']) + 1)
            ax.plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
            ax.plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: mAP
        ax = axes[1, 1]
        if history.get('train_map') and history.get('val_map'):
            epochs = range(1, len(history['train_map']) + 1)
            ax.plot(epochs, history['train_map'], 'b-', label='Train mAP', linewidth=2)
            ax.plot(epochs, history['val_map'], 'r-', label='Val mAP', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.set_title('mAP Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to: {plot_path}")
        plt.close()
        
        # Check for issues in curves
        print("\nCurve Analysis:")
        if history.get('val_loss'):
            val_losses = [v for v in history['val_loss'] if v is not None]
            if len(val_losses) > 5:
                recent_trend = np.mean(val_losses[-5:]) - np.mean(val_losses[-10:-5]) if len(val_losses) > 10 else 0
                if recent_trend > 0.01:
                    print("  ⚠️  Val loss is increasing (diverging)")
                elif abs(recent_trend) < 0.001:
                    print("  ⚠️  Val loss is flat (not learning)")
                else:
                    print("  ✓ Val loss is decreasing (good)")
    
    def check_data_pipeline(self) -> Dict:
        """Check data pipeline: RGB channels, normalization, augmentations."""
        print("\n" + "="*80)
        print("4. DATA PIPELINE CHECKS")
        print("="*80)
        
        results = {}
        
        # Load checkpoint to get config
        config = None
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            if 'hyper_parameters' in checkpoint:
                config = checkpoint['hyper_parameters']
        except Exception as e:
            print(f"Warning: Could not load checkpoint config: {e}")
        
        # For BigEarthNetv2_0, also try to reconstruct ILMConfiguration
        ilm_config = None
        if self.model_type == 'bigearthnet' and config:
            try:
                ilm_config = self._reconstruct_ilm_config(config)
            except Exception as e:
                print(f"Warning: Could not reconstruct ILMConfiguration: {e}")
        
        # Check RGB channels
        print("\nRGB Channel Configuration:")
        num_channels = None
        if config and 'channels' in config:
            num_channels = config['channels']
        elif ilm_config and hasattr(ilm_config, 'channels'):
            num_channels = ilm_config.channels
        
        results['num_channels'] = num_channels
        
        if num_channels is not None:
            print(f"  Total channels: {num_channels}")
            if num_channels >= 3:
                if self.model_type == 'multimodal':
                    rgb_bands = config.get('rgb_band_names', ['B04', 'B03', 'B02'])
                    print(f"  ✓ RGB channels (0-2): {rgb_bands}")
                    print(f"  → Multimodal model extracts RGB as: rgb_data = x[:, :3, :, :]")
                    results['rgb_bands'] = rgb_bands
                    
                    # Check band configuration
                    bandconfig = config.get('bandconfig', 's2s1')
                    use_s1 = config.get('use_s1', False)
                    resnet_enabled = config.get('resnet_enabled', True)
                    print(f"  Band configuration: {bandconfig}")
                    print(f"  S1 usage: {'ENABLED' if use_s1 else 'DISABLED'}")
                    print(f"  ResNet branch: {'ENABLED' if resnet_enabled else 'DISABLED'}")
                    results['bandconfig'] = bandconfig
                    results['use_s1'] = use_s1
                    results['resnet_enabled'] = resnet_enabled
                else:
                    print(f"  ✓ RGB channels (0-2) should be: B04 (Red), B03 (Green), B02 (Blue)")
                    print(f"  → BigEarthNetv2_0 model uses all channels: model(x)")
                    print(f"  → First 3 channels are RGB: B04, B03, B02")
            else:
                print(f"  ⚠️  Only {num_channels} channels - may not have RGB")
        else:
            print("  ⚠️  Could not determine channel configuration from checkpoint")
        
        # Check normalization
        print("\nNormalization:")
        print("  Expected: Normalize transform with mean/std from BENv2DataModule")
        print("  → Check train_transform in data module for Normalize transform")
        print("  → Mean/std should match pretraining if using pretrained backbones")
        
        # Check augmentations
        print("\nAugmentations:")
        print("  Expected transforms:")
        print("    - RandomHorizontalFlip")
        print("    - RandomVerticalFlip")
        print("    - Normalize (with BENv2 mean/std)")
        print("  → Check if augmentations are too heavy (causing underfitting)")
        print("  → Check if augmentations are too weak (causing overfitting)")
        
        # Try to instantiate data module to check
        try:
            print("\nAttempting to instantiate data module...")
            hostname, data_dirs = get_benv2_dir_dict(config_path=str(self.config_path) if self.config_path else None)
            data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
            
            if config or ilm_config:
                # Get channels from config or ilm_config
                channels = None
                if config and 'channels' in config:
                    channels = config['channels']
                elif ilm_config and hasattr(ilm_config, 'channels'):
                    channels = ilm_config.channels
                elif config and 'bandconfig' in config:
                    # Try to infer from bandconfig
                    _, channels = get_bands(config['bandconfig'])
                
                if channels is None:
                    channels = 12  # Default fallback
                
                hparams = {
                    'batch_size': 32,
                    'workers': 4,
                    'channels': channels,
                }
                dm = default_dm(hparams, data_dirs, img_size=120)
                
                print(f"  ✓ Data module created successfully")
                print(f"  Train transform: {type(dm.train_transform)}")
                transforms_info = []
                if hasattr(dm.train_transform, 'transforms'):
                    print(f"  Transforms:")
                    for i, t in enumerate(dm.train_transform.transforms):
                        transform_name = type(t).__name__
                        print(f"    {i+1}. {transform_name}")
                        transforms_info.append(transform_name)
                        if isinstance(t, torch.nn.Module):
                            if hasattr(t, 'mean') and hasattr(t, 'std'):
                                mean_vals = t.mean[:3] if len(t.mean) >= 3 else t.mean
                                std_vals = t.std[:3] if len(t.std) >= 3 else t.std
                                print(f"       Mean: {mean_vals} (first 3)")
                                print(f"       Std:  {std_vals} (first 3)")
                                results['normalization_mean'] = mean_vals.tolist() if hasattr(mean_vals, 'tolist') else list(mean_vals)
                                results['normalization_std'] = std_vals.tolist() if hasattr(std_vals, 'tolist') else list(std_vals)
                
                results['transforms'] = transforms_info
        except Exception as e:
            print(f"  ⚠️  Could not instantiate data module: {e}")
        
        return results
    
    def plot_per_class_metrics(self, results_df: pd.DataFrame, output_dir: Path):
        """Plot per-class metrics."""
        if results_df is None:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Per-Class Metrics', fontsize=16, fontweight='bold')
        
        # Sort by F1 score
        results_sorted = results_df.sort_values('F1', ascending=True)
        
        # Plot 1: F1 scores
        ax = axes[0, 0]
        ax.barh(range(len(results_sorted)), results_sorted['F1'].values)
        ax.set_yticks(range(len(results_sorted)))
        ax.set_yticklabels(results_sorted['Class'].values, fontsize=8)
        ax.set_xlabel('F1 Score')
        ax.set_title('Per-Class F1 Scores')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='F1=0.5')
        ax.legend()
        
        # Plot 2: Precision
        ax = axes[0, 1]
        ax.barh(range(len(results_sorted)), results_sorted['Precision'].values)
        ax.set_yticks(range(len(results_sorted)))
        ax.set_yticklabels(results_sorted['Class'].values, fontsize=8)
        ax.set_xlabel('Precision')
        ax.set_title('Per-Class Precision')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Recall
        ax = axes[1, 0]
        ax.barh(range(len(results_sorted)), results_sorted['Recall'].values)
        ax.set_yticks(range(len(results_sorted)))
        ax.set_yticklabels(results_sorted['Class'].values, fontsize=8)
        ax.set_xlabel('Recall')
        ax.set_title('Per-Class Recall')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Support
        ax = axes[1, 1]
        ax.barh(range(len(results_sorted)), results_sorted['Support'].values)
        ax.set_yticks(range(len(results_sorted)))
        ax.set_yticklabels(results_sorted['Class'].values, fontsize=8)
        ax.set_xlabel('Support (Number of Samples)')
        ax.set_title('Per-Class Support')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_path = output_dir / "per_class_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class metrics plot to: {plot_path}")
        plt.close()
    
    def save_diagnostics_summary(self, output_dir: Path, history: Dict, performance_results: Dict, 
                                  data_pipeline_results: Dict, per_class_results: Optional[pd.DataFrame]):
        """Save comprehensive diagnostics summary to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save hyperparameters
        hparams_path = output_dir / "hyperparameters.yaml"
        with open(hparams_path, 'w') as f:
            yaml.dump(self.hparams, f, default_flow_style=False, sort_keys=False)
        print(f"Saved hyperparameters to: {hparams_path}")
        
        # Save model configuration
        config_path = output_dir / "model_config.yaml"
        config_to_save = {
            'model_type': self.model_type,
            'model_config': self.model_config,
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
        print(f"Saved model configuration to: {config_path}")
        
        # Save training history CSV
        if history:
            history_df = pd.DataFrame(history)
            history_path = output_dir / "training_history.csv"
            history_df.to_csv(history_path, index=False)
            print(f"Saved training history to: {history_path}")
        
        # Save performance summary
        summary = {
            'checkpoint_path': str(self.checkpoint_path),
            'model_type': self.model_type,
            'run_directory': str(self.run_dir) if self.run_dir else None,
            'lightning_logs': str(self.lightning_logs_dir) if self.lightning_logs_dir else None,
            'performance': performance_results,
            'data_pipeline': data_pipeline_results,
            'timestamp': datetime.now().isoformat(),
        }
        
        summary_path = output_dir / "diagnostics_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved diagnostics summary to: {summary_path}")
        
        # Save text report
        report_path = output_dir / "diagnostics_report.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING DIAGNOSTICS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Model Type: {self.model_type}\n")
            if self.run_dir:
                f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETERS\n")
            f.write("="*80 + "\n")
            for key, value in self.hparams.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            if performance_results:
                f.write("="*80 + "\n")
                f.write("PERFORMANCE SUMMARY\n")
                f.write("="*80 + "\n")
                for key, value in performance_results.items():
                    if key != 'diagnosis':
                        f.write(f"  {key}: {value}\n")
                if 'diagnosis' in performance_results:
                    f.write("\nDiagnosis:\n")
                    for msg in performance_results['diagnosis']:
                        f.write(f"  {msg}\n")
                f.write("\n")
            
            if data_pipeline_results:
                f.write("="*80 + "\n")
                f.write("DATA PIPELINE\n")
                f.write("="*80 + "\n")
                for key, value in data_pipeline_results.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"Saved diagnostics report to: {report_path}")
    
    def run_all_diagnostics(self, output_dir: Path, data_module=None, num_samples: int = 1000):
        """Run all diagnostic checks."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("TRAINING DIAGNOSTICS")
        print("="*80)
        
        # Load training history
        history = self.load_training_history()
        
        # 1. Train vs Val performance
        performance_results = self.check_train_vs_val_performance(history)
        
        # 2. Per-class confusion matrix
        results_df = self.compute_confusion_matrix(
            data_module=data_module,
            num_samples=num_samples
        )
        
        if results_df is not None:
            self.plot_per_class_metrics(results_df, output_dir)
            # Save results to CSV
            csv_path = output_dir / "per_class_metrics.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Saved per-class metrics to: {csv_path}")
        
        # 3. Training curves
        self.plot_training_curves(history, output_dir)
        
        # 4. Data pipeline checks
        data_pipeline_results = self.check_data_pipeline()
        
        # 5. Save comprehensive diagnostics
        self.save_diagnostics_summary(
            output_dir=output_dir,
            history=history,
            performance_results=performance_results,
            data_pipeline_results=data_pipeline_results,
            per_class_results=results_df,
        )
        
        print("\n" + "="*80)
        print("DIAGNOSTICS COMPLETE")
        print("="*80)
        print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Training Diagnostics - Analyzes training runs from checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diagnose_training.py --checkpoint ./checkpoints/model.ckpt
  python diagnose_training.py --checkpoint ./ckpt_logs/run_name/checkpoints/best.ckpt --output-dir ./my_diagnostics
  python diagnose_training.py --checkpoint model.ckpt --num-samples 2000
  python diagnose_training.py --checkpoint best.ckpt --config-path ../train_scripts/config.yaml
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file (.ckpt) - REQUIRED. Can use resolve_checkpoint_path logic (e.g., "best", "last", or relative path)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: ./diagnostics/<checkpoint_name>)')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples for confusion matrix computation (default: 1000)')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to config YAML for data directories (optional, aligns with train_multimodal.py)')
    
    args = parser.parse_args()
    
    # Resolve checkpoint path (supports "best", "last", or file paths)
    checkpoint_path = args.checkpoint
    if checkpoint_path.lower() in ["best", "last"]:
        # Try to resolve from run directory if we're in one
        # Otherwise, user should provide full path
        checkpoint_path = checkpoint_path.lower()
    
    # Create diagnostics object (auto-discovers logs from checkpoint location)
    try:
        config_path_obj = Path(args.config_path) if args.config_path else None
        diag = TrainingDiagnostics(
            checkpoint_path=checkpoint_path,
            config_path=config_path_obj
        )
    except FileNotFoundError as e:
        parser.error(str(e))
    
    # Set default output directory based on checkpoint name if not provided
    if args.output_dir is None:
        checkpoint_name = diag.checkpoint_path.stem
        if diag.run_dir:
            # Use run directory name if available
            output_dir = diag.run_dir / "diagnostics"
        else:
            output_dir = Path("./diagnostics") / checkpoint_name
    else:
        output_dir = Path(args.output_dir)
    
    # Try to create data module from checkpoint
    data_module = None
    try:
        checkpoint = torch.load(diag.checkpoint_path, map_location='cpu')
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters'].copy()
            
            # Get data directories (aligned with train_multimodal.py)
            hostname, data_dirs = get_benv2_dir_dict(config_path=args.config_path)
            data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
            hparams['batch_size'] = 32  # Use smaller batch for diagnostics
            hparams['workers'] = 4
            
            # Ensure channels is set
            if 'channels' not in hparams:
                # Try to infer from bandconfig or architecture
                if 'bandconfig' in hparams:
                    _, channels = get_bands(hparams['bandconfig'])
                    hparams['channels'] = channels
                else:
                    # Default based on model type
                    if diag.model_type == 'multimodal':
                        use_s1 = hparams.get('use_s1', False)
                        hparams['channels'] = 14 if use_s1 else 12  # RGB(3) + S2_nonRGB(9) + S1(2) or RGB(3) + S2_nonRGB(9)
                    else:
                        hparams['channels'] = 12  # Default fallback
            
            # For multimodal, register band configuration
            if diag.model_type == 'multimodal' and 'bandconfig' in hparams:
                bandconfig = hparams['bandconfig']
                use_s1 = hparams.get('use_s1', False)
                
                # Reconstruct band order (aligned with train_multimodal.py)
                rgb_bands = ["B04", "B03", "B02"]
                s2_bands = STANDARD_BANDS.get("S2", STANDARD_BANDS.get("s2_full", []))
                s1_bands = STANDARD_BANDS.get("S1", ["VV", "VH"])
                full_s2_non_rgb = [b for b in s2_bands if b not in rgb_bands]
                
                if bandconfig == "rgb":
                    multimodal_bands = rgb_bands
                elif use_s1:
                    multimodal_bands = rgb_bands + full_s2_non_rgb + s1_bands
                else:
                    multimodal_bands = rgb_bands + full_s2_non_rgb
                
                # Register with BENv2DataSet (same as train_multimodal.py)
                num_channels = len(multimodal_bands)
                STANDARD_BANDS[num_channels] = multimodal_bands
                STANDARD_BANDS["multimodal"] = multimodal_bands
                BENv2DataSet.channel_configurations[num_channels] = multimodal_bands
                BENv2DataSet.avail_chan_configs[num_channels] = f"Multimodal ({bandconfig})"
            
            data_module = default_dm(hparams, data_dirs, img_size=120)
            print(f"Created data module with {hparams.get('channels', 'unknown')} channels")
    except Exception as e:
        print(f"Warning: Could not create data module: {e}")
        import traceback
        traceback.print_exc()
        print("  Per-class confusion matrix will be skipped.")
        print("  Other diagnostics will still run.")
    
    # Run diagnostics
    diag.run_all_diagnostics(
        output_dir=output_dir,
        data_module=data_module,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
