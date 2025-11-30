"""
Multimodal Interpretability: Modality Occlusion Sensitivity Analysis.

This script analyzes which modality (RGB/DINOv3 vs Non-RGB/ResNet) contributes
most to predictions for each class by systematically occluding each modality
and measuring the drop in prediction confidence.
"""

import sys
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import typer
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import NEW_LABELS  # noqa: E402
from scripts.diagnose.model_utils import load_model_and_infer  # noqa: E402

__author__ = "Modality Importance Analysis Tool"


def extract_multimodal_backbones(model) -> Dict[str, Optional[nn.Module]]:
    """
    Extract DINOv3 and ResNet backbones from multimodal model.
    
    Returns:
        Dictionary with 'dinov3' and 'resnet' keys
    """
    backbones = {'dinov3': None, 'resnet': None}
    
    # Check for multimodal model structure
    if hasattr(model, 'model'):
        if hasattr(model.model, 'dinov3_backbone'):
            dinov3_backbone = model.model.dinov3_backbone
            if hasattr(dinov3_backbone, 'backbone'):
                backbones['dinov3'] = dinov3_backbone.backbone
            else:
                backbones['dinov3'] = dinov3_backbone
        
        if hasattr(model.model, 'resnet_backbone'):
            backbones['resnet'] = model.model.resnet_backbone
    
    return backbones


def occlude_modality(
    model: nn.Module,
    x: torch.Tensor,
    modality: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Occlude a specific modality by zeroing out the corresponding channels.
    
    Args:
        model: Multimodal model
        x: Input tensor (B, C, H, W)
        modality: 'rgb' to occlude RGB (DINOv3), 'non_rgb' to occlude non-RGB (ResNet)
        device: Device to run on
    
    Returns:
        Modified input tensor with occluded modality
    """
    x_occluded = x.clone()
    
    if modality == 'rgb':
        # Occlude RGB channels (first 3 channels: B04, B03, B02)
        x_occluded[:, :3, :, :] = 0.0
    elif modality == 'non_rgb':
        # Occlude non-RGB channels (channels 3 onwards)
        if x.shape[1] > 3:
            x_occluded[:, 3:, :, :] = 0.0
    else:
        raise ValueError(f"Unknown modality: {modality}. Use 'rgb' or 'non_rgb'")
    
    return x_occluded


def compute_modality_importance(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute modality importance by measuring confidence drop when occluding each modality.
    
    Args:
        model: Multimodal model
        dataloader: DataLoader for test/validation set
        device: Device to run on
        n_samples: Number of samples to process (None = all)
    
    Returns:
        Dictionary with:
        - 'rgb_importance': (N, 19) confidence drop when RGB is occluded
        - 'non_rgb_importance': (N, 19) confidence drop when non-RGB is occluded
        - 'labels': (N, 19) true labels
    """
    model.eval()
    
    rgb_drops = []
    non_rgb_drops = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if n_samples is not None and batch_idx * x.shape[0] >= n_samples:
                break
            
            x = x.to(device)
            y = y.to(device)
            
            # Get baseline predictions (full input)
            logits_full = model(x)
            probs_full = torch.sigmoid(logits_full)
            
            # Occlude RGB and get predictions
            x_no_rgb = occlude_modality(model, x, 'rgb', device)
            logits_no_rgb = model(x_no_rgb)
            probs_no_rgb = torch.sigmoid(logits_no_rgb)
            
            # Occlude non-RGB and get predictions
            x_no_non_rgb = occlude_modality(model, x, 'non_rgb', device)
            logits_no_non_rgb = model(x_no_non_rgb)
            probs_no_non_rgb = torch.sigmoid(logits_no_non_rgb)
            
            # Compute confidence drop for true labels only
            # Drop = baseline_confidence - occluded_confidence
            # Positive drop means the modality is important (occlusion hurts)
            
            # For each sample and class, compute drop only if label is positive
            for i in range(x.shape[0]):
                sample_labels = y[i].cpu().numpy()  # (19,)
                sample_probs_full = probs_full[i].cpu().numpy()  # (19,)
                sample_probs_no_rgb = probs_no_rgb[i].cpu().numpy()  # (19,)
                sample_probs_no_non_rgb = probs_no_non_rgb[i].cpu().numpy()  # (19,)
                
                # Only compute for positive labels
                positive_mask = sample_labels > 0.5
                
                # RGB importance: drop when RGB is occluded
                rgb_drop = np.zeros(19)
                rgb_drop[positive_mask] = (
                    sample_probs_full[positive_mask] - sample_probs_no_rgb[positive_mask]
                )
                
                # Non-RGB importance: drop when non-RGB is occluded
                non_rgb_drop = np.zeros(19)
                non_rgb_drop[positive_mask] = (
                    sample_probs_full[positive_mask] - sample_probs_no_non_rgb[positive_mask]
                )
                
                rgb_drops.append(rgb_drop)
                non_rgb_drops.append(non_rgb_drop)
                all_labels.append(sample_labels)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    return {
        'rgb_importance': np.array(rgb_drops),
        'non_rgb_importance': np.array(non_rgb_drops),
        'labels': np.array(all_labels),
    }


def visualize_modality_importance(
    importance_data: Dict[str, np.ndarray],
    class_names: list,
    output_path: str,
) -> None:
    """
    Create stacked bar chart showing relative contribution of RGB vs Non-RGB per class.
    
    Args:
        importance_data: Dictionary from compute_modality_importance
        class_names: List of 19 class names
        output_path: Path to save visualization
    """
    rgb_importance = importance_data['rgb_importance']  # (N, 19)
    non_rgb_importance = importance_data['non_rgb_importance']  # (N, 19)
    labels = importance_data['labels']  # (N, 19)
    
    # Compute mean importance per class (only for positive samples)
    n_classes = len(class_names)
    mean_rgb = np.zeros(n_classes)
    mean_non_rgb = np.zeros(n_classes)
    
    for i in range(n_classes):
        # Only consider samples where this class is positive
        positive_mask = labels[:, i] > 0.5
        if positive_mask.sum() > 0:
            mean_rgb[i] = rgb_importance[positive_mask, i].mean()
            mean_non_rgb[i] = non_rgb_importance[positive_mask, i].mean()
    
    # Normalize to show relative contribution
    total_importance = mean_rgb + mean_non_rgb
    # Avoid division by zero
    total_importance[total_importance == 0] = 1.0
    rgb_ratio = mean_rgb / total_importance
    non_rgb_ratio = mean_non_rgb / total_importance
    
    # Sort by total importance
    total_sorted = total_importance.copy()
    sorted_indices = np.argsort(total_sorted)[::-1]
    
    sorted_rgb_ratio = rgb_ratio[sorted_indices]
    sorted_non_rgb_ratio = non_rgb_ratio[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_total = total_sorted[sorted_indices]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    
    x_pos = np.arange(len(sorted_classes))
    width = 0.8
    
    # Create stacked bars
    bars1 = ax.barh(x_pos, sorted_rgb_ratio, width, label='RGB (DINOv3)', 
                    color='#2E86AB', alpha=0.8, edgecolor='black')
    bars2 = ax.barh(x_pos, sorted_non_rgb_ratio, width, left=sorted_rgb_ratio,
                    label='Non-RGB (ResNet)', color='#A23B72', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (rgb_r, non_rgb_r, total) in enumerate(zip(sorted_rgb_ratio, sorted_non_rgb_ratio, sorted_total)):
        # Label with percentages
        if rgb_r > 0.1:  # Only label if significant
            ax.text(rgb_r / 2, i, f'{rgb_r*100:.0f}%', 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        if non_rgb_r > 0.1:
            ax.text(rgb_r + non_rgb_r / 2, i, f'{non_rgb_r*100:.0f}%',
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Add total importance on the right
        ax.text(1.05, i, f'{total:.3f}', va='center', fontsize=9)
    
    # Customize axes
    ax.set_yticks(x_pos)
    ax.set_yticklabels(sorted_classes, fontsize=10)
    ax.set_xlabel('Relative Modality Contribution (Normalized)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Modality Importance per Class\n'
        '(RGB/DINOv3 vs Non-RGB/ResNet Contribution)',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlim([0, 1.2])
    
    # Add legend
    ax.legend(loc='lower right', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add summary
    rgb_dominant = (rgb_ratio > 0.6).sum()
    non_rgb_dominant = (non_rgb_ratio > 0.6).sum()
    balanced = ((rgb_ratio >= 0.4) & (rgb_ratio <= 0.6)).sum()
    
    stats_text = (
        f'RGB-dominant classes (>60%): {rgb_dominant}\n'
        f'Non-RGB-dominant classes (>60%): {non_rgb_dominant}\n'
        f'Balanced classes (40-60%): {balanced}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved modality importance chart to: {output_path}")
    plt.close()


def main(
    model_path: str = typer.Option(
        ...,
        help="Path to multimodal model checkpoint (must have DINOv3 and ResNet backbones)"
    ),
    model_type: Optional[str] = typer.Option(
        None,
        help="Model type (auto-detected if not provided)"
    ),
    threshold: float = typer.Option(
        0.5,
        help="Classification threshold (not used for importance, but for model loading)"
    ),
    bs: int = typer.Option(
        32,
        help="Batch size"
    ),
    workers: int = typer.Option(
        8,
        help="Number of workers"
    ),
    test_run: bool = typer.Option(
        False,
        help="Run with fewer batches for quick testing"
    ),
    n_samples: Optional[int] = typer.Option(
        None,
        help="Number of samples to analyze (None = all)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        help="Output directory for visualizations"
    ),
    split: str = typer.Option(
        "test",
        help="Dataset split to use: 'test' or 'validation'"
    ),
):
    """
    Analyze modality importance in multimodal models.
    
    This script systematically occludes RGB (DINOv3) and non-RGB (ResNet) channels
    and measures the drop in prediction confidence to determine which modality
    is more important for each class.
    """
    # Setup output directory
    if output_dir is None:
        model_path_obj = Path(model_path)
        if model_path_obj.is_file():
            output_dir = model_path_obj.parent
        else:
            output_dir = Path(".") / "modality_importance"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Modality Importance Analysis")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Split: {split}")
    print("=" * 80 + "\n")
    
    # Load model and setup dataloader
    from scripts.diagnose.model_utils import _detect_model_type
    import lightning.pytorch as pl
    from scripts.utils import get_benv2_dir_dict, default_dm
    from configilm.extra.BENv2_utils import resolve_data_dir
    from multimodal.lightning_module import MultiModalLightningModule
    import torch
    
    if model_type is None:
        model_type = _detect_model_type(model_path)
    
    if model_type != "checkpoint_multimodal":
        raise ValueError(
            "This analysis requires a multimodal model with both DINOv3 and ResNet backbones. "
            f"Detected model type: {model_type}"
        )
    
    # Load model
    print("Loading multimodal model...")
    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    hparams = checkpoint.get('hyper_parameters', {})
    
    # Setup datamodule (simplified - you may need to adjust based on your checkpoint)
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
    
    hparams_dm = {
        "batch_size": bs,
        "workers": workers,
        "channels": hparams.get('channels', 14),
        "bandconfig": hparams.get('bandconfig', 's2s1'),
    }
    dm = default_dm(hparams_dm, data_dirs, 120)
    
    if split == "validation":
        dm.setup(stage="fit")
        dataloader = dm.val_dataloader()
    else:
        dm.setup(stage="test")
        dataloader = dm.test_dataloader()
    
    # Load model (simplified - adjust as needed)
    # You'll need to construct the config properly based on your checkpoint
    model = MultiModalLightningModule.load_from_checkpoint(str(ckpt_path))
    model.eval()
    
    device = next(model.parameters()).device
    
    # Compute modality importance
    print("\nComputing modality importance...")
    print("This may take a while as we need to run inference multiple times...")
    importance_data = compute_modality_importance(
        model,
        dataloader,
        device,
        n_samples=n_samples if not test_run else 100,
    )
    
    # Generate visualization
    print("\nGenerating visualization...")
    output_path = output_dir / f"modality_importance_{Path(model_path).stem}.png"
    visualize_modality_importance(
        importance_data,
        NEW_LABELS,
        str(output_path),
    )
    
    # Save numerical results
    import pandas as pd
    rgb_importance = importance_data['rgb_importance']
    non_rgb_importance = importance_data['non_rgb_importance']
    
    # Compute mean per class
    labels = importance_data['labels']
    mean_rgb = np.array([rgb_importance[labels[:, i] > 0.5, i].mean() 
                         if (labels[:, i] > 0.5).sum() > 0 else 0.0 
                         for i in range(19)])
    mean_non_rgb = np.array([non_rgb_importance[labels[:, i] > 0.5, i].mean()
                              if (labels[:, i] > 0.5).sum() > 0 else 0.0
                              for i in range(19)])
    
    total = mean_rgb + mean_non_rgb
    total[total == 0] = 1.0
    rgb_ratio = mean_rgb / total
    non_rgb_ratio = mean_non_rgb / total
    
    results_df = pd.DataFrame({
        'Class': NEW_LABELS,
        'RGB_Importance': mean_rgb,
        'NonRGB_Importance': mean_non_rgb,
        'RGB_Ratio': rgb_ratio,
        'NonRGB_Ratio': non_rgb_ratio,
        'Total_Importance': total,
    })
    results_df = results_df.sort_values('Total_Importance', ascending=False)
    
    results_path = output_dir / f"modality_importance_{Path(model_path).stem}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved numerical results to: {results_path}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    typer.run(main)

