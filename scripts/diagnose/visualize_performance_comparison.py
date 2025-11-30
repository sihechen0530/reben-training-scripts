"""
Performance Comparison Visualizations for BigEarthNet v2.0 models.

Generates two key visualizations:
1. Per-Class Delta Bar Chart - Shows AP improvement per class
2. Precision-Recall Scatter Plot - Shows per-class AP comparison

These visualizations help identify where model improvements come from
and which classes benefit most from the new approach.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import NEW_LABELS  # noqa: E402
from scripts.diagnose.model_utils import load_model_and_infer  # noqa: E402

__author__ = "Performance Comparison Visualization Tool"


def compute_per_class_ap(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Compute Average Precision (AP) for each class.
    
    Args:
        labels: (N, 19) binary labels
        probabilities: (N, 19) prediction probabilities
    
    Returns:
        (19,) array of AP scores per class
    """
    n_classes = labels.shape[1]
    ap_scores = np.zeros(n_classes)
    
    for i in range(n_classes):
        y_true = labels[:, i]
        y_scores = probabilities[:, i]
        
        # Skip if no positive samples
        if y_true.sum() == 0:
            ap_scores[i] = 0.0
            continue
        
        try:
            ap_scores[i] = average_precision_score(y_true, y_scores)
        except Exception:
            ap_scores[i] = 0.0
    
    return ap_scores


def visualize_per_class_delta(
    ap_baseline: np.ndarray,
    ap_yours: np.ndarray,
    class_names: list,
    output_path: str,
    baseline_name: str = "Baseline",
    your_name: str = "Your Model",
) -> None:
    """
    Create per-class delta bar chart showing AP improvement.
    
    Args:
        ap_baseline: (19,) AP scores for baseline model
        ap_yours: (19,) AP scores for your model
        class_names: List of 19 class names
        output_path: Path to save visualization
        baseline_name: Name for baseline model
        your_name: Name for your model
    """
    delta_ap = ap_yours - ap_baseline
    
    # Sort by delta (largest improvement first)
    sorted_indices = np.argsort(delta_ap)[::-1]
    sorted_deltas = delta_ap[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_baseline = ap_baseline[sorted_indices]
    sorted_yours = ap_yours[sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color bars: green for positive, red for negative
    colors = ['green' if d > 0 else 'red' for d in sorted_deltas]
    
    # Create bars
    bars = ax.barh(range(len(sorted_classes)), sorted_deltas, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, delta, baseline, yours) in enumerate(zip(bars, sorted_deltas, sorted_baseline, sorted_yours)):
        # Label with delta
        label_x = delta + (0.01 if delta >= 0 else -0.01)
        ax.text(label_x, i, f'{delta:+.3f}\n({baseline:.3f}→{yours:.3f})',
                va='center', ha='left' if delta >= 0 else 'right', fontsize=9)
    
    # Customize axes
    ax.set_yticks(range(len(sorted_classes)))
    ax.set_yticklabels(sorted_classes, fontsize=10)
    ax.set_xlabel(f'Δ AP = AP({your_name}) - AP({baseline_name})', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Per-Class Average Precision Improvement\n'
        f'{your_name} vs {baseline_name}',
        fontsize=14,
        fontweight='bold'
    )
    
    # Add zero line
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add summary statistics
    mean_delta = delta_ap.mean()
    positive_count = (delta_ap > 0).sum()
    total_count = len(delta_ap)
    
    stats_text = (
        f'Mean Δ AP: {mean_delta:+.4f}\n'
        f'Classes improved: {positive_count}/{total_count} ({100*positive_count/total_count:.1f}%)'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved per-class delta chart to: {output_path}")
    plt.close()


def visualize_pr_scatter(
    ap_baseline: np.ndarray,
    ap_yours: np.ndarray,
    class_names: list,
    output_path: str,
    baseline_name: str = "Baseline",
    your_name: str = "Your Model",
) -> None:
    """
    Create Precision-Recall scatter plot showing per-class AP comparison.
    
    Args:
        ap_baseline: (19,) AP scores for baseline model
        ap_yours: (19,) AP scores for your model
        class_names: List of 19 class names
        output_path: Path to save visualization
        baseline_name: Name for baseline model
        your_name: Name for your model
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot
    scatter = ax.scatter(ap_baseline, ap_yours, s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add class labels
    for i, class_name in enumerate(class_names):
        ax.annotate(
            class_name,
            (ap_baseline[i], ap_yours[i]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Add diagonal line (y = x)
    min_ap = min(ap_baseline.min(), ap_yours.min())
    max_ap = max(ap_baseline.max(), ap_yours.max())
    ax.plot([min_ap, max_ap], [min_ap, max_ap], 'r--', linewidth=2, label='y=x (equal performance)', alpha=0.7)
    
    # Color points above/below diagonal
    above_line = ap_yours > ap_baseline
    below_line = ap_yours < ap_baseline
    
    if above_line.any():
        ax.scatter(ap_baseline[above_line], ap_yours[above_line], 
                  s=150, alpha=0.8, color='green', edgecolors='black', linewidth=1.5,
                  label=f'Improved ({above_line.sum()} classes)', zorder=3)
    
    if below_line.any():
        ax.scatter(ap_baseline[below_line], ap_yours[below_line],
                  s=150, alpha=0.8, color='red', edgecolors='black', linewidth=1.5,
                  label=f'Degraded ({below_line.sum()} classes)', zorder=3)
    
    # Customize axes
    ax.set_xlabel(f'{baseline_name} Average Precision', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{your_name} Average Precision', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Per-Class Average Precision Comparison\n'
        f'Points above diagonal = {your_name} outperforms {baseline_name}',
        fontsize=14,
        fontweight='bold'
    )
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([min_ap - 0.05, max_ap + 0.05])
    ax.set_ylim([min_ap - 0.05, max_ap + 0.05])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10)
    
    # Add statistics
    mean_baseline = ap_baseline.mean()
    mean_yours = ap_yours.mean()
    mean_improvement = mean_yours - mean_baseline
    
    # Count hard classes (low baseline AP) that improved
    hard_threshold = 0.3  # Classes with AP < 0.3 are considered "hard"
    hard_classes = ap_baseline < hard_threshold
    hard_improved = (ap_yours > ap_baseline) & hard_classes
    
    stats_text = (
        f'Mean AP: {baseline_name}={mean_baseline:.4f}, {your_name}={mean_yours:.4f}\n'
        f'Mean improvement: {mean_improvement:+.4f}\n'
        f'Hard classes improved: {hard_improved.sum()}/{hard_classes.sum()} '
        f'({100*hard_improved.sum()/hard_classes.sum() if hard_classes.sum() > 0 else 0:.1f}%)'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved PR scatter plot to: {output_path}")
    plt.close()


def main(
    baseline_model: str = typer.Option(
        ...,
        help="Path to baseline model (checkpoint file or HuggingFace model name)"
    ),
    your_model: str = typer.Option(
        ...,
        help="Path to your model (checkpoint file or HuggingFace model name)"
    ),
    baseline_type: Optional[str] = typer.Option(
        None,
        help="Model type for baseline (auto-detected if not provided)"
    ),
    your_model_type: Optional[str] = typer.Option(
        None,
        help="Model type for your model (auto-detected if not provided)"
    ),
    baseline_name: str = typer.Option(
        "Baseline",
        help="Display name for baseline model"
    ),
    your_name: str = typer.Option(
        "Your Model",
        help="Display name for your model"
    ),
    threshold: float = typer.Option(
        0.5,
        help="Classification threshold for predictions"
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
    Generate performance comparison visualizations between two models.
    
    Creates:
    1. Per-Class Delta Bar Chart - Shows AP improvement per class
    2. Precision-Recall Scatter Plot - Shows per-class AP comparison
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(".") / "performance_comparison"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Performance Comparison Analysis")
    print("=" * 80)
    print(f"Baseline Model: {baseline_model}")
    print(f"Your Model: {your_model}")
    print(f"Split: {split}")
    print("=" * 80 + "\n")
    
    # Load baseline model predictions
    print("Loading baseline model predictions...")
    _, probs_baseline, labels, _ = load_model_and_infer(
        model_path=baseline_model,
        model_type=baseline_type,
        threshold=threshold,
        bs=bs,
        workers=workers,
        test_run=test_run,
        split=split,
    )
    
    # Load your model predictions
    print("\nLoading your model predictions...")
    _, probs_yours, labels_yours, _ = load_model_and_infer(
        model_path=your_model,
        model_type=your_model_type,
        threshold=threshold,
        bs=bs,
        workers=workers,
        test_run=test_run,
        split=split,
    )
    
    # Verify labels match
    if not np.array_equal(labels, labels_yours):
        print("Warning: Labels don't match between models. Using baseline labels.")
        labels = labels  # Use baseline labels
    
    # Compute per-class AP
    print("\nComputing per-class Average Precision...")
    ap_baseline = compute_per_class_ap(labels, probs_baseline)
    ap_yours = compute_per_class_ap(labels, probs_yours)
    
    print(f"\nBaseline Mean AP: {ap_baseline.mean():.4f}")
    print(f"Your Model Mean AP: {ap_yours.mean():.4f}")
    print(f"Mean Improvement: {ap_yours.mean() - ap_baseline.mean():+.4f}")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    # 1. Per-class delta bar chart
    delta_path = output_dir / f"per_class_delta_{baseline_name.replace(' ', '_')}_vs_{your_name.replace(' ', '_')}.png"
    visualize_per_class_delta(
        ap_baseline,
        ap_yours,
        NEW_LABELS,
        str(delta_path),
        baseline_name=baseline_name,
        your_name=your_name,
    )
    
    # 2. PR scatter plot
    scatter_path = output_dir / f"pr_scatter_{baseline_name.replace(' ', '_')}_vs_{your_name.replace(' ', '_')}.png"
    visualize_pr_scatter(
        ap_baseline,
        ap_yours,
        NEW_LABELS,
        str(scatter_path),
        baseline_name=baseline_name,
        your_name=your_name,
    )
    
    # Save numerical results
    results_path = output_dir / f"per_class_ap_comparison.csv"
    import pandas as pd
    results_df = pd.DataFrame({
        'Class': NEW_LABELS,
        f'AP_{baseline_name}': ap_baseline,
        f'AP_{your_name}': ap_yours,
        'Delta_AP': ap_yours - ap_baseline,
    })
    results_df = results_df.sort_values('Delta_AP', ascending=False)
    results_df.to_csv(results_path, index=False)
    print(f"Saved numerical results to: {results_path}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    typer.run(main)

