"""
Calculate AUC and plot ROC curves for a BigEarthNet v2.0 model.

Supports three model types:
1. HuggingFace pretrained models (model_name string)
2. Local BigEarthNet checkpoints (.ckpt file)
3. Local Multimodal checkpoints (.ckpt file)
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import NEW_LABELS  # noqa: E402
from scripts.diagnose.model_utils import load_model_and_infer  # noqa: E402

__author__ = "BIFOLD/RSiM TU Berlin"


def calculate_auc_and_plot_roc(
    probabilities: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[str] = None,
    class_names: Optional[list] = None,
    show_plot: bool = True,
):
    """
    Calculate AUC scores and plot ROC curves for multilabel classification.
    
    Args:
        probabilities: Predicted probabilities, shape (n_samples, n_classes)
        labels: True labels, shape (n_samples, n_classes)
        output_path: Optional path to save the plot
        class_names: Optional list of class names (default: NEW_LABELS)
        show_plot: Whether to display the plot
    """
    if class_names is None:
        class_names = NEW_LABELS
    
    n_classes = probabilities.shape[1]
    if len(class_names) != n_classes:
        print(f"Warning: Number of class names ({len(class_names)}) doesn't match "
              f"number of classes ({n_classes}). Using generic names.")
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    # Calculate per-class AUC
    per_class_auc = []
    fpr_dict = {}
    tpr_dict = {}
    
    for i in range(n_classes):
        y_true = labels[:, i]
        y_score = probabilities[:, i]
        
        # Skip if class has no positive samples
        if np.sum(y_true) == 0:
            print(f"Warning: Class {i} ({class_names[i]}) has no positive samples. Skipping.")
            per_class_auc.append(np.nan)
            continue
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        per_class_auc.append(roc_auc)
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
    
    # Calculate macro-averaged AUC
    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    macro_auc = np.mean(valid_aucs) if valid_aucs else np.nan
    
    # Calculate micro-averaged AUC (one-vs-rest)
    try:
        micro_auc = roc_auc_score(labels, probabilities, average='micro')
    except Exception as e:
        print(f"Warning: Could not calculate micro-averaged AUC: {e}")
        micro_auc = np.nan
    
    # Calculate sample-averaged AUC
    try:
        sample_auc = roc_auc_score(labels, probabilities, average='samples')
    except Exception as e:
        print(f"Warning: Could not calculate sample-averaged AUC: {e}")
        sample_auc = np.nan
    
    # Print results
    print("\n" + "=" * 80)
    print("AUC SCORES")
    print("=" * 80)
    print(f"Macro-averaged AUC: {macro_auc:.4f}")
    print(f"Micro-averaged AUC: {micro_auc:.4f}")
    print(f"Sample-averaged AUC: {sample_auc:.4f}")
    print("\nPer-class AUC:")
    for i, (auc_score, name) in enumerate(zip(per_class_auc, class_names)):
        if not np.isnan(auc_score):
            print(f"  {i:2d}. {name:30s}: {auc_score:.4f}")
        else:
            print(f"  {i:2d}. {name:30s}: N/A (no positive samples)")
    print("=" * 80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ROC Curves', fontsize=16, fontweight='bold')
    
    # Plot 1: All classes together (without legend to avoid clutter)
    ax1 = axes[0, 0]
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    for i in range(n_classes):
        if i in fpr_dict and not np.isnan(per_class_auc[i]):
            ax1.plot(
                fpr_dict[i],
                tpr_dict[i],
                color=colors[i],
                lw=1.5,
                alpha=0.7,
            )
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'All Classes (Macro AUC = {macro_auc:.3f})', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Top 10 classes by AUC
    ax2 = axes[0, 1]
    valid_indices = [(i, auc) for i, auc in enumerate(per_class_auc) if not np.isnan(auc)]
    valid_indices.sort(key=lambda x: x[1], reverse=True)
    top_10_indices = [i for i, _ in valid_indices[:10]]
    
    for idx in top_10_indices:
        if idx in fpr_dict:
            ax2.plot(
                fpr_dict[idx],
                tpr_dict[idx],
                lw=2,
                label=f'{class_names[idx]} (AUC = {per_class_auc[idx]:.3f})'
            )
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('Top 10 Classes by AUC', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Plot 3: Bottom 10 classes by AUC
    ax3 = axes[1, 0]
    bottom_10_indices = [i for i, _ in valid_indices[-10:]]
    
    for idx in bottom_10_indices:
        if idx in fpr_dict:
            ax3.plot(
                fpr_dict[idx],
                tpr_dict[idx],
                lw=2,
                label=f'{class_names[idx]} (AUC = {per_class_auc[idx]:.3f})'
            )
    ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate', fontsize=12)
    ax3.set_ylabel('True Positive Rate', fontsize=12)
    ax3.set_title('Bottom 10 Classes by AUC', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(alpha=0.3)
    
    # Plot 4: AUC scores bar chart
    ax4 = axes[1, 1]
    valid_aucs_with_names = [(class_names[i], auc) for i, auc in enumerate(per_class_auc) if not np.isnan(auc)]
    valid_aucs_with_names.sort(key=lambda x: x[1], reverse=True)
    names_sorted = [name for name, _ in valid_aucs_with_names]
    aucs_sorted = [auc for _, auc in valid_aucs_with_names]
    
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(aucs_sorted)))
    bars = ax4.barh(range(len(aucs_sorted)), aucs_sorted, color=colors_bar)
    ax4.set_yticks(range(len(names_sorted)))
    ax4.set_yticklabels(names_sorted, fontsize=8)
    ax4.set_xlabel('AUC Score', fontsize=12)
    ax4.set_title('AUC Scores by Class (Sorted)', fontsize=12, fontweight='bold')
    ax4.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax4.axvline(x=macro_auc, color='g', linestyle='--', alpha=0.5, label=f'Macro Avg ({macro_auc:.3f})')
    ax4.set_xlim([0.0, 1.0])
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, auc_val) in enumerate(zip(bars, aucs_sorted)):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{auc_val:.3f}', ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return {
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'sample_auc': sample_auc,
        'per_class_auc': per_class_auc,
    }


def main(
    model_path: str = typer.Option(..., "--model-path", "-m", help="Path to model checkpoint or HuggingFace model name"),
    model_type: Optional[str] = typer.Option(None, "--model-type", "-t", 
                                             help="Model type: 'hf_pretrained', 'checkpoint_bigearthnet', or 'checkpoint_multimodal'. "
                                                  "If None, will be auto-detected."),
    architecture: Optional[str] = typer.Option(None, "--architecture", "-a", 
                                               help="Model architecture (e.g., 'dinov3-base', 'resnet101'). "
                                                    "If None, will be inferred from checkpoint."),
    bandconfig: Optional[str] = typer.Option(None, "--bandconfig", "-b",
                                             help="Band configuration (e.g., 'rgb', 's2', 'all'). "
                                                  "If None, will be inferred from checkpoint."),
    use_s1: Optional[bool] = typer.Option(None, "--use-s1/--no-use-s1",
                                         help="Whether to use S1 data (for multimodal models). "
                                              "If None, will be inferred from checkpoint."),
    dinov3_model_name: Optional[str] = typer.Option(None, "--dinov3-model-name",
                                                     help="DINOv3 HuggingFace model name. "
                                                          "If None, will be inferred."),
    threshold: float = typer.Option(0.5, "--threshold",
                                   help="Threshold for binary predictions (not used for AUC calculation, but for reference)"),
    test_run: bool = typer.Option(False, "--test-run/--no-test-run",
                                  help="Run with fewer batches for quick testing"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o",
                                              help="Path to save the ROC curve plot. If None, will save to default location based on model path."),
    show_plot: bool = typer.Option(True, "--show-plot/--no-show-plot",
                                   help="Whether to display the plot"),
):
    """
    Calculate AUC scores and plot ROC curves for a BigEarthNet v2.0 model.
    
    Example usage:
        python diagnose/plot_roc_curve.py --model-path ./checkpoints/model.ckpt
        python diagnose/plot_roc_curve.py --model-path ./checkpoints/model.ckpt --output roc_curves.png
    """
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    
    print("=" * 80)
    print("ROC CURVE ANALYSIS")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Model type: {model_type or 'auto-detect'}")
    print(f"Threshold: {threshold}")
    print("=" * 80)
    
    # Load model and run inference
    print("\nLoading model and running inference...")
    predictions, probabilities, labels, _ = load_model_and_infer(
        model_path=model_path,
        model_type=model_type,
        architecture=architecture,
        bandconfig=bandconfig,
        use_s1=use_s1,
        dinov3_model_name=dinov3_model_name,
        threshold=threshold,
        test_run=test_run,
    )
    
    print(f"\nInference complete:")
    print(f"  Samples: {len(predictions)}")
    print(f"  Classes: {probabilities.shape[1]}")
    
    # Generate default output path if not provided
    if output_path is None:
        model_path_obj = Path(model_path)
        # Create output directory based on model path
        if model_path_obj.is_file():
            # If it's a checkpoint file, save in the same directory
            output_dir = model_path_obj.parent
            output_filename = f"{model_path_obj.stem}_roc_curves.png"
        else:
            # If it's a HuggingFace model name, save in current directory
            output_dir = Path(".")
            # Sanitize model name for filename
            safe_name = str(model_path).replace("/", "_").replace("\\", "_")
            output_filename = f"{safe_name}_roc_curves.png"
        
        output_path = str(output_dir / output_filename)
        print(f"\nUsing default output path: {output_path}")
    
    # Calculate AUC and plot ROC curves
    results = calculate_auc_and_plot_roc(
        probabilities=probabilities,
        labels=labels,
        output_path=output_path,
        class_names=NEW_LABELS,
        show_plot=show_plot,
    )
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    typer.run(main)

