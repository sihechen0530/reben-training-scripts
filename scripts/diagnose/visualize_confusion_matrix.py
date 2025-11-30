"""
Visualize mAP (mean Average Precision) per class for a BigEarthNet v2.0 model.

Supports three model types:
1. HuggingFace pretrained models (model_name string)
2. Local BigEarthNet checkpoints (.ckpt file)
3. Local Multimodal checkpoints (.ckpt file)
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer
import matplotlib.pyplot as plt
import seaborn as sns
import lightning.pytorch as pl
import torch
from sklearn.metrics import average_precision_score, f1_score
from scipy import stats

# Optional: adjustText for better label positioning
try:
    from adjustText import adjust_text
    ADJUSTTEXT_AVAILABLE = True
except ImportError:
    ADJUSTTEXT_AVAILABLE = False

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import NEW_LABELS  # noqa: E402
from scripts.diagnose.model_utils import load_model_and_infer  # noqa: E402

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def _infer_dinov3_model_from_checkpoint(checkpoint_path: str) -> str:
    """Infer DINOv3 model name from checkpoint."""
    from pathlib import Path
    import torch
    
    ckpt_path = Path(checkpoint_path)
    filename = ckpt_path.name.lower()
    
    if "dinov3-large" in filename or "dinov3-l" in filename:
        return "facebook/dinov3-vitl16-pretrain-lvd1689m"
    elif "dinov3-base" in filename or "dinov3-b" in filename:
        return "facebook/dinov3-vitb16-pretrain-lvd1689m"
    elif "dinov3-small" in filename or "dinov3-s" in filename:
        return "facebook/dinov3-vits16-pretrain-lvd1689m"
    elif "dinov3-giant" in filename or "dinov3-g" in filename:
        return "facebook/dinov3-vitg16-pretrain-lvd1689m"
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            if 'dinov3_model_name' in hparams:
                return hparams['dinov3_model_name']
    except:
        pass
    
    return "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _detect_model_type(model_path: str) -> str:
    """Detect model type: 'hf_pretrained', 'checkpoint_bigearthnet', or 'checkpoint_multimodal'."""
    model_path_str = str(model_path).lower()
    
    # Check if it's a HuggingFace model name (contains '/' and doesn't end with .ckpt)
    if '/' in model_path and not model_path.endswith('.ckpt') and not Path(model_path).exists():
        return 'hf_pretrained'
    
    # Check if it's a file path
    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        # If it doesn't exist as file, assume it's HF model name
        return 'hf_pretrained'
    
    # Try to load checkpoint and detect type
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Check for multimodal indicators in state_dict or hyperparameters
        state_dict = checkpoint.get('state_dict', {})
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Check for multimodal model keys
        multimodal_keys = ['model.dinov3_backbone', 'model.resnet_backbone', 'model.fusion']
        has_multimodal_keys = any(key in str(state_dict.keys()) for key in multimodal_keys)
        
        # Check hyperparameters
        if 'use_s1' in hparams or 'fusion_type' in hparams:
            return 'checkpoint_multimodal'
        
        if has_multimodal_keys:
            return 'checkpoint_multimodal'
        
        return 'checkpoint_bigearthnet'
    except:
        # Default to BigEarthNet if cannot determine
        return 'checkpoint_bigearthnet'

def compute_per_class_map(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Compute mean Average Precision (mAP) for each class.
    
    Args:
        labels: (N, 19) binary labels
        probabilities: (N, 19) prediction probabilities
    
    Returns:
        (19,) array of mAP scores per class
    """
    n_classes = labels.shape[1]
    map_scores = np.zeros(n_classes)
    
    for i in range(n_classes):
        y_true = labels[:, i]
        y_scores = probabilities[:, i]
        
        # Skip if no positive samples
        if y_true.sum() == 0:
            map_scores[i] = 0.0
            continue
        
        try:
            map_scores[i] = average_precision_score(y_true, y_scores)
        except Exception:
            map_scores[i] = 0.0
    
    return map_scores


def visualize_map_heatmap(
    map_scores: np.ndarray,
    class_names: list,
    output_path: str = "map_heatmap.png",
    figsize: tuple = (12, 10),
) -> None:
    """
    Visualize mAP scores per class as a heatmap.
    
    Args:
        map_scores: (19,) array of mAP scores
        class_names: List of 19 class names
        output_path: Path to save visualization
        figsize: Figure size
    """
    # Sort by mAP (highest first)
    sorted_indices = np.argsort(map_scores)[::-1]
    sorted_map = map_scores[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    
    # Reshape for heatmap (single row, 19 columns)
    map_data = sorted_map.reshape(1, -1)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = sns.heatmap(
        map_data,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'mAP Score'},
        yticklabels=["mAP"],
        xticklabels=sorted_classes,
    )
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    ax.set_title(
        f"Mean Average Precision (mAP) per Class\n"
        f"Mean mAP: {map_scores.mean():.4f}, Macro mAP: {map_scores.mean():.4f}",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved mAP heatmap to: {output_path}")
    plt.close()


def visualize_map_bar_chart(
    map_scores: np.ndarray,
    class_names: list,
    output_path: str = "map_bar_chart.png",
    figsize: tuple = (16, 8),
) -> None:
    """
    Create bar chart showing mAP per class.
    
    Args:
        map_scores: (19,) array of mAP scores
        class_names: List of 19 class names
        output_path: Path to save visualization
        figsize: Figure size
    """
    # Sort by mAP (highest first)
    sorted_indices = np.argsort(map_scores)[::-1]
    sorted_map = map_scores[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color bars based on mAP value
    colors = plt.cm.RdYlGn(sorted_map)
    
    bars = ax.barh(range(len(sorted_classes)), sorted_map, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, map_val) in enumerate(zip(bars, sorted_map)):
        ax.text(map_val + 0.01, i, f'{map_val:.4f}',
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # Customize axes
    ax.set_yticks(range(len(sorted_classes)))
    ax.set_yticklabels(sorted_classes, fontsize=10)
    ax.set_xlabel('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'mAP per Class\nMean mAP: {map_scores.mean():.4f}',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlim([0, 1.1])
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add mean line
    mean_map = map_scores.mean()
    ax.axvline(mean_map, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean: {mean_map:.4f}')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved mAP bar chart to: {output_path}")
    plt.close()


def compute_class_frequencies(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute class frequencies (count and percentage of positive samples).
    
    Args:
        labels: (N, 19) binary labels
    
    Returns:
        frequencies: (19,) array of positive sample counts per class
        percentages: (19,) array of positive sample percentages per class
    """
    n_samples = labels.shape[0]
    frequencies = labels.sum(axis=0)  # Count of positive samples per class
    percentages = (frequencies / n_samples) * 100  # Percentage of positive samples
    return frequencies, percentages


def visualize_scarcity_vs_map_absolute(
    map_scores: np.ndarray,
    frequencies: np.ndarray,
    class_names: list,
    output_path: str = "scarcity_vs_map.png",
    model_name: str = "Model",
    figsize: tuple = (12, 10),
) -> None:
    """
    Visualize the relationship between class scarcity (frequency) and absolute mAP.
    
    Fallback visualization when no baseline is provided.
    
    Args:
        map_scores: (19,) array of mAP scores per class
        frequencies: (19,) array of positive sample counts per class
        class_names: List of 19 class names
        output_path: Path to save visualization
        model_name: Name for the model
        figsize: Figure size
    """
    # Use log scale for frequencies (Power Law distribution)
    log_frequencies = np.log10(frequencies + 1)  # +1 to avoid log(0)
    
    # Calculate correlation
    valid_mask = frequencies > 0
    if valid_mask.sum() > 1:
        corr, p_value = stats.pearsonr(log_frequencies[valid_mask], map_scores[valid_mask])
    else:
        corr, p_value = 0.0, 1.0
    
    # Create single plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color points by mAP value
    scatter = ax.scatter(
        frequencies,
        map_scores,
        s=200,
        alpha=0.7,
        c=map_scores,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        edgecolors='black',
        linewidths=2,
        zorder=3
    )
    
    # Identify outliers and interesting points to label
    # Top 3 mAP, Bottom 3 mAP, and interesting outliers
    top_map = np.argsort(map_scores)[-3:][::-1]  # Top 3
    bottom_map = np.argsort(map_scores)[:3]  # Bottom 3
    
    # Interesting outliers: high mAP with low frequency, or low mAP with low frequency
    interesting_indices = set(top_map.tolist() + bottom_map.tolist())
    
    # Add "Marine waters" and "Beaches, dunes, sands" if they're interesting
    try:
        marine_idx = class_names.index("Marine waters")
        beaches_idx = class_names.index("Beaches, dunes, sands")
        interesting_indices.add(marine_idx)
        interesting_indices.add(beaches_idx)
    except ValueError:
        pass
    
    # Label only interesting points
    labels_to_add = []
    for i in interesting_indices:
        if i < len(class_names):
            labels_to_add.append((frequencies[i], map_scores[i], class_names[i]))
    
    # Try to use adjustText for better label positioning, fallback to manual
    if ADJUSTTEXT_AVAILABLE:
        texts = []
        for x, y, label in labels_to_add:
            texts.append(ax.text(x, y, label, fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                         alpha=0.9, edgecolor='black', linewidth=1)))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                   expand_points=(1.2, 1.2), expand_text=(1.2, 1.2))
    else:
        # Fallback: manual positioning with offset
        for x, y, label in labels_to_add:
            # Offset based on quadrant
            if y > map_scores.mean():
                xytext = (10, 10)  # Top-right
            else:
                xytext = (10, -15)  # Bottom-right
            
            ax.annotate(
                label,
                (x, y),
                xytext=xytext,
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.9, edgecolor='black', linewidth=1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)
            )
    
    # Fit linear regression for trend line
    if valid_mask.sum() > 1:
        x_fit = log_frequencies[valid_mask]
        y_fit = map_scores[valid_mask]
        slope, intercept, r_value, p_value_fit, std_err = stats.linregress(x_fit, y_fit)
        
        # Generate trend line
        x_trend = np.logspace(np.log10(frequencies[valid_mask].min()), 
                             np.log10(frequencies[valid_mask].max()), 100)
        y_trend = slope * np.log10(x_trend) + intercept
        ax.plot(x_trend, y_trend, 'b--', linewidth=3, alpha=0.7, 
                label=f'Trend Line (r={corr:.3f}, p={p_value:.3e})', zorder=2)
    
    # Customize axes
    ax.set_xlabel('Number of Positive Samples (Log Scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Mean Average Precision (mAP)', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Class Scarcity vs. mAP\n'
        f'{model_name} Performance Across Class Frequencies',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)
    
    # Set limits
    x_min = frequencies[frequencies > 0].min() * 0.7
    x_max = frequencies.max() * 1.3
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([-0.05, 1.05])
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('mAP Score', fontsize=11)
    
    # Add interpretation box (top-left)
    interpretation_text = (
        'Interpretation:\n'
        '• Rare classes (left) vs Common classes (right)\n'
        '• Higher mAP = Better performance\n'
        '• Positive correlation = Common classes perform better'
    )
    ax.text(0.02, 0.98, interpretation_text, 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # Add statistics (bottom-right)
    mean_map = map_scores.mean()
    high_performers = (map_scores > 0.7).sum()
    low_performers = (map_scores < 0.3).sum()
    
    stats_text = (
        f'Statistics:\n'
        f'Mean mAP: {mean_map:.4f}\n'
        f'High (mAP > 0.7): {high_performers}/19 classes\n'
        f'Low (mAP < 0.3): {low_performers}/19 classes'
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, 
                    edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scarcity vs mAP visualization to: {output_path}")
    print(f"  Correlation: r={corr:.3f}, p={p_value:.3e}")
    print(f"  Mean mAP: {mean_map:.4f}")
    if corr > 0.3:
        print(f"  → Strong positive correlation: Common classes perform significantly better")
    elif corr > 0.1:
        print(f"  → Moderate positive correlation: Some bias toward common classes")
    elif corr < -0.1:
        print(f"  → Negative correlation: Rare classes perform relatively better (unusual!)")
    else:
        print(f"  → Weak correlation: Performance relatively uniform across frequencies")
    plt.close()


def visualize_scarcity_vs_map(
    baseline_map_scores: np.ndarray,
    your_map_scores: np.ndarray,
    frequencies: np.ndarray,
    class_names: list,
    output_path: str = "scarcity_vs_map.png",
    baseline_name: str = "Baseline",
    your_name: str = "Your Model",
    figsize: tuple = (12, 10),
) -> None:
    """
    Visualize the relationship between class scarcity (frequency) and Delta mAP.
    
    Publication-worthy visualization showing:
    - X-axis: Log Class Frequency (Power Law distribution)
    - Y-axis: Δ mAP = mAP(Your Model) - mAP(Baseline)
    - Highlights outliers and shows if DINOv3 improves few-shot learning
    
    Args:
        baseline_map_scores: (19,) array of mAP scores for baseline model
        your_map_scores: (19,) array of mAP scores for your model
        frequencies: (19,) array of positive sample counts per class
        class_names: List of 19 class names
        output_path: Path to save visualization
        baseline_name: Name for baseline model
        your_name: Name for your model
        figsize: Figure size
    """
    # Calculate Delta mAP
    delta_map = your_map_scores - baseline_map_scores
    
    # Use log scale for frequencies (Power Law distribution)
    log_frequencies = np.log10(frequencies + 1)  # +1 to avoid log(0)
    
    # Calculate correlation
    valid_mask = frequencies > 0
    if valid_mask.sum() > 1:
        corr, p_value = stats.pearsonr(log_frequencies[valid_mask], delta_map[valid_mask])
    else:
        corr, p_value = 0.0, 1.0
    
    # Create single plot (removed redundant percentage plot)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color points: green for improvement, red for degradation
    colors = ['green' if d > 0 else 'red' for d in delta_map]
    alphas = [0.8 if abs(d) > 0.05 else 0.4 for d in delta_map]  # Highlight significant changes
    
    # Plot scatter
    scatter = ax.scatter(
        frequencies,
        delta_map,
        s=200,
        alpha=alphas,
        c=colors,
        edgecolors='black',
        linewidths=2,
        zorder=3
    )
    
    # Identify outliers and interesting points to label
    # Top 3 improvements, Bottom 3 degradations, and interesting outliers
    top_improvements = np.argsort(delta_map)[-3:][::-1]  # Top 3
    top_degradations = np.argsort(delta_map)[:3]  # Bottom 3
    
    # Interesting outliers: high improvement with low frequency, or high degradation with low frequency
    interesting_indices = set(top_improvements.tolist() + top_degradations.tolist())
    
    # Add "Marine waters" and "Beaches, dunes, sands" if they're interesting
    try:
        marine_idx = class_names.index("Marine waters")
        beaches_idx = class_names.index("Beaches, dunes, sands")
        interesting_indices.add(marine_idx)
        interesting_indices.add(beaches_idx)
    except ValueError:
        pass
    
    # Label only interesting points
    labels_to_add = []
    for i in interesting_indices:
        if i < len(class_names):
            labels_to_add.append((frequencies[i], delta_map[i], class_names[i]))
    
    # Try to use adjustText for better label positioning, fallback to manual
    if ADJUSTTEXT_AVAILABLE:
        texts = []
        for x, y, label in labels_to_add:
            texts.append(ax.text(x, y, label, fontsize=10, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                         alpha=0.9, edgecolor='black', linewidth=1)))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
                   expand_points=(1.2, 1.2), expand_text=(1.2, 1.2))
    else:
        # Fallback: manual positioning with offset
        # Note: Install adjustText for better label positioning: pip install adjusttext
        for x, y, label in labels_to_add:
            # Offset based on quadrant
            if y > 0:
                xytext = (10, 10)  # Top-right
            else:
                xytext = (10, -15)  # Bottom-right
            
            ax.annotate(
                label,
                (x, y),
                xytext=xytext,
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.9, edgecolor='black', linewidth=1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.5)
            )
    
    # Fit linear regression for trend line
    trend_interpretation = ""
    trend_color = 'gray'
    if valid_mask.sum() > 1:
        x_fit = log_frequencies[valid_mask]
        y_fit = delta_map[valid_mask]
        slope, intercept, r_value, p_value_fit, std_err = stats.linregress(x_fit, y_fit)
        
        # Generate trend line
        x_trend = np.logspace(np.log10(frequencies[valid_mask].min()), 
                             np.log10(frequencies[valid_mask].max()), 100)
        y_trend = slope * np.log10(x_trend) + intercept
        ax.plot(x_trend, y_trend, 'b--', linewidth=3, alpha=0.7, 
                label=f'Trend Line (r={corr:.3f}, p={p_value:.3e})', zorder=2)
        
        # Interpret the trend
        if abs(corr) < 0.1:
            trend_interpretation = "Flat trend: Uniform improvement across all classes"
            trend_color = 'gray'
        elif corr < -0.1:  # Negative correlation
            trend_interpretation = "↓ Negative correlation: 'Holy Grail' - Helps rare classes MORE!"
            trend_color = 'green'
        else:  # Positive correlation
            trend_interpretation = "↑ Positive correlation: Mainly improves common classes"
            trend_color = 'orange'
    
    # Add horizontal line at Y=0 (baseline performance)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2.5, alpha=0.6, 
              label='Baseline Performance (Y=0)', zorder=1)
    
    # Customize axes
    ax.set_xlabel('Number of Positive Samples (Log Scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'Δ mAP = mAP({your_name}) - mAP({baseline_name})', fontsize=13, fontweight='bold')
    ax.set_title(
        f'Class Scarcity vs. Model Improvement\n'
        f'DINOv3 Few-Shot Learning Performance on Rare Classes',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)
    
    # Set limits
    x_min = frequencies[frequencies > 0].min() * 0.7
    x_max = frequencies.max() * 1.3
    y_range = delta_map.max() - delta_map.min()
    y_margin = y_range * 0.15  # More margin for annotations
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([delta_map.min() - y_margin, delta_map.max() + y_margin])
    
    # Add interpretation box (top-left)
    interpretation_text = (
        'Interpretation:\n'
        '• Positive Δ = Improvement over baseline\n'
        '• Negative Δ = Degradation from baseline\n'
        '• Rare classes (left) vs Common classes (right)'
    )
    ax.text(0.02, 0.98, interpretation_text, 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    # Add trend interpretation (top-right)
    if trend_interpretation:
        ax.text(0.98, 0.98, trend_interpretation, 
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               color=trend_color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor=trend_color, linewidth=2))
    
    # Add statistics (bottom-right)
    mean_improvement = delta_map.mean()
    improved_classes = (delta_map > 0).sum()
    degraded_classes = (delta_map < 0).sum()
    
    stats_text = (
        f'Statistics:\n'
        f'Mean Δ mAP: {mean_improvement:+.4f}\n'
        f'Improved: {improved_classes}/19 classes\n'
        f'Degraded: {degraded_classes}/19 classes'
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, 
                    edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Higher DPI for publication
    print(f"Saved scarcity vs Delta mAP visualization to: {output_path}")
    print(f"  Correlation: r={corr:.3f}, p={p_value:.3e}")
    if trend_interpretation:
        print(f"  Trend interpretation: {trend_interpretation}")
    print(f"  Mean improvement: {mean_improvement:+.4f}")
    print(f"  Improved classes: {improved_classes}/19")
    print(f"  Degraded classes: {degraded_classes}/19")
    if corr < -0.1:
        print(f"  ✓ 'Holy Grail' detected: Negative correlation indicates DINOv3 helps rare classes MORE!")
    elif corr > 0.1:
        print(f"  ⚠ Positive correlation: Model mainly improves common classes (may indicate overfitting to head)")
    else:
        print(f"  → Flat trend: Uniform improvement across all classes")
    plt.close()


def main(
    model_path: str = typer.Option(..., help="Path to checkpoint file or HuggingFace model name (e.g., 'hackelle/resnet18-all-v0.1.1')"),
    baseline_model_path: Optional[str] = typer.Option(None, help="Path to baseline model for comparison. If provided, will create Delta mAP visualization."),
    model_type: Optional[str] = typer.Option(None, help="Model type: 'hf_pretrained', 'checkpoint_bigearthnet', or 'checkpoint_multimodal'. Auto-detected if not provided."),
    baseline_model_type: Optional[str] = typer.Option(None, help="Baseline model type"),
    architecture: Optional[str] = typer.Option(None, help="Model architecture (auto-detected if not provided)"),
    bandconfig: Optional[str] = typer.Option(None, help="Band configuration (auto-detected if not provided)"),
    use_s1: Optional[bool] = typer.Option(None, "--use-s1/--no-use-s1", help="Whether to include S1 (for multimodal models)"),
    seed: int = typer.Option(42, help="Random seed"),
    lr: float = typer.Option(0.001, help="Learning rate (for model initialization)"),
    drop_rate: float = typer.Option(0.15, help="Dropout rate"),
    drop_path_rate: float = typer.Option(0.15, help="Drop path rate"),
    warmup: int = typer.Option(1000, help="Warmup steps"),
    bs: int = typer.Option(32, help="Batch size"),
    workers: int = typer.Option(8, help="Number of workers"),
    threshold: float = typer.Option(0.5, help="Classification threshold"),
    test_run: bool = typer.Option(False, help="Run with fewer batches for quick testing"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory for visualizations. If None, will save to same directory as model."),
    dinov3_model_name: Optional[str] = typer.Option(None, help="DINOv3 HuggingFace model name"),
    baseline_name: str = typer.Option("Baseline", help="Name for baseline model in visualization"),
    your_name: str = typer.Option("Your Model", help="Name for your model in visualization"),
):
    """
    Visualize mAP (mean Average Precision) per class for a BigEarthNet v2.0 model.
    
    Supports three model types:
    1. HuggingFace pretrained: Provide model name (e.g., 'hackelle/resnet18-all-v0.1.1')
    2. BigEarthNet checkpoint: Provide path to .ckpt file
    3. Multimodal checkpoint: Provide path to .ckpt file (auto-detected)
    """
    # Generate default output directory if not provided
    if output_dir is None:
        model_path_obj = Path(model_path)
        # Create output directory based on model path
        if model_path_obj.is_file():
            # If it's a checkpoint file, save in the same directory
            output_dir = str(model_path_obj.parent)
        else:
            # If it's a HuggingFace model name, save in current directory
            output_dir = "."
        print(f"\nUsing default output directory: {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and run inference
    predictions, probabilities, labels, _ = load_model_and_infer(
        model_path=model_path,
        model_type=model_type,
        architecture=architecture,
        bandconfig=bandconfig,
        use_s1=use_s1,
        seed=seed,
        lr=lr,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        warmup=warmup,
        bs=bs,
        workers=workers,
        test_run=test_run,
        dinov3_model_name=dinov3_model_name,
        threshold=threshold,
    )
    
    # Compute per-class mAP
    print("\nComputing per-class mAP...")
    your_map_scores = compute_per_class_map(labels, probabilities)
    
    # Load baseline model if provided
    baseline_map_scores = None
    if baseline_model_path:
        print(f"\nLoading baseline model: {baseline_model_path}")
        _, baseline_probs, baseline_labels, _ = load_model_and_infer(
            model_path=baseline_model_path,
            model_type=baseline_model_type,
            architecture=architecture,
            bandconfig=bandconfig,
            use_s1=use_s1,
            seed=seed,
            lr=lr,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            warmup=warmup,
            bs=bs,
            workers=workers,
            test_run=test_run,
            dinov3_model_name=dinov3_model_name,
            threshold=threshold,
        )
        baseline_map_scores = compute_per_class_map(baseline_labels, baseline_probs)
        print("✓ Baseline mAP computed")
    
    # Use your model's map_scores for single-model visualizations
    map_scores = your_map_scores
    
    # Compute class frequencies (scarcity)
    print("Computing class frequencies (scarcity)...")
    frequencies, percentages = compute_class_frequencies(labels)
    
    # Print summary statistics
    mean_map = map_scores.mean()
    print(f"\nMean mAP (macro-averaged): {mean_map:.4f}")
    print(f"mAP range: [{map_scores.min():.4f}, {map_scores.max():.4f}]")
    print(f"Classes with mAP > 0.5: {(map_scores > 0.5).sum()}/19")
    print(f"Classes with mAP > 0.7: {(map_scores > 0.7).sum()}/19")
    
    # Print frequency statistics
    print(f"\nClass frequency statistics:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Frequency range: [{frequencies.min()}, {frequencies.max()}]")
    print(f"  Percentage range: [{percentages.min():.2f}%, {percentages.max():.2f}%]")
    print(f"  Most common class: {NEW_LABELS[np.argmax(frequencies)]} ({frequencies.max()} samples, {percentages.max():.2f}%)")
    print(f"  Rarest class: {NEW_LABELS[np.argmin(frequencies)]} ({frequencies.min()} samples, {percentages.min():.2f}%)")
    
    # Generate output filenames based on model path
    model_path_obj = Path(model_path)
    if model_path_obj.is_file():
        base_name = model_path_obj.stem
    else:
        # Sanitize model name for filename
        base_name = str(model_path).replace("/", "_").replace("\\", "_")
    
    map_heatmap_path = str(output_dir / f"{base_name}_map_heatmap.png")
    map_bar_path = str(output_dir / f"{base_name}_map_bar_chart.png")
    
    if baseline_map_scores is not None:
        scarcity_map_path = str(output_dir / f"{base_name}_scarcity_vs_delta_map.png")
    else:
        scarcity_map_path = str(output_dir / f"{base_name}_scarcity_vs_map.png")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_map_heatmap(
        map_scores,
        NEW_LABELS,
        output_path=map_heatmap_path,
    )
    
    visualize_map_bar_chart(
        map_scores,
        NEW_LABELS,
        output_path=map_bar_path,
    )
    
    # Scarcity visualization: Delta mAP if baseline provided, otherwise absolute mAP
    if baseline_map_scores is not None:
        print("\nGenerating publication-worthy Delta mAP visualization...")
        visualize_scarcity_vs_map(
            baseline_map_scores,
            your_map_scores,
            frequencies,
            NEW_LABELS,
            output_path=scarcity_map_path,
            baseline_name=baseline_name,
            your_name=your_name,
        )
    else:
        print("\n⚠ No baseline model provided. For Delta mAP visualization, use --baseline-model-path")
        print("   Generating absolute mAP vs frequency visualization instead...")
        visualize_scarcity_vs_map_absolute(
            map_scores,
            frequencies,
            NEW_LABELS,
            output_path=scarcity_map_path,
            model_name=your_name,
        )
    
    # Save numerical results
    import pandas as pd
    results_data = {
        'Class': NEW_LABELS,
        'mAP': map_scores,
        'Frequency': frequencies,
        'Percentage': percentages,
    }
    
    if baseline_map_scores is not None:
        results_data['Baseline_mAP'] = baseline_map_scores
        results_data['Delta_mAP'] = your_map_scores - baseline_map_scores
        sort_column = 'Delta_mAP'
    else:
        sort_column = 'mAP'
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(sort_column, ascending=False)
    results_csv = output_dir / f"{base_name}_map_scores.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved mAP scores and frequencies to: {results_csv}")
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - mAP heatmap: {map_heatmap_path}")
    print(f"  - mAP bar chart: {map_bar_path}")
    print(f"  - Scarcity vs mAP: {scarcity_map_path}")
    print(f"  - mAP scores CSV: {results_csv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    typer.run(main)


