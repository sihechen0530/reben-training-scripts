"""
Visualize confusion matrices for a BigEarthNet v2.0 model.

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
from sklearn.metrics import multilabel_confusion_matrix, f1_score

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

def visualize_confusion_matrices(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    output_path: str = "confusion_matrix.png",
    figsize: tuple = (20, 16),
) -> None:
    """Visualize per-class confusion matrices for multilabel classification."""
    n_classes = len(class_names)
    mcm = multilabel_confusion_matrix(labels, predictions)

    n_cols = 5
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_classes > 1 else [axes]

    for i in range(n_classes):
        ax = axes[i]
        cm = mcm[i]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
        )

        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        ax.set_title(f"{class_names[i]}\nP={precision:.3f}, R={recall:.3f}, F1={f1:.3f}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for i in range(n_classes, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved confusion matrix visualization to: {output_path}")
    plt.close()


def visualize_summary_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    output_path: str = "confusion_summary.png",
) -> None:
    """Create summary matrices with confusion counts and precision/recall/F1 per class."""
    mcm = multilabel_confusion_matrix(labels, predictions)
    metrics = []
    for i, class_name in enumerate(class_names):
        tn, fp, fn, tp = mcm[i].ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics.append(
            {
                "Class": class_name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )
    
    # Calculate micro and macro F1 scores
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    print(f"\nOverall F1 Scores:")
    print(f"  Micro-averaged F1: {micro_f1:.4f}")
    print(f"  Macro-averaged F1: {macro_f1:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    counts_data = np.array([[m["TP"], m["FP"], m["FN"], m["TN"]] for m in metrics])
    sns.heatmap(
        counts_data,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=axes[0],
        xticklabels=["TP", "FP", "FN", "TN"],
        yticklabels=[m["Class"] for m in metrics],
    )
    axes[0].set_title("Confusion Matrix Counts per Class", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Class", fontsize=12)

    metrics_data = np.array([[m["Precision"], m["Recall"], m["F1"]] for m in metrics])
    sns.heatmap(
        metrics_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=axes[1],
        vmin=0,
        vmax=1,
        xticklabels=["Precision", "Recall", "F1"],
        yticklabels=[m["Class"] for m in metrics],
    )
    axes[1].set_title(
        f"Performance Metrics per Class (Micro F1: {micro_f1:.3f}, Macro F1: {macro_f1:.3f})",
        fontsize=14,
        fontweight="bold"
    )
    axes[1].set_ylabel("Class", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary visualization to: {output_path}")
    plt.close()


def main(
    model_path: str = typer.Option(..., help="Path to checkpoint file or HuggingFace model name (e.g., 'hackelle/resnet18-all-v0.1.1')"),
    model_type: Optional[str] = typer.Option(None, help="Model type: 'hf_pretrained', 'checkpoint_bigearthnet', or 'checkpoint_multimodal'. Auto-detected if not provided."),
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
):
    """
    Visualize confusion matrices for a BigEarthNet v2.0 model.
    
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
    
    # Generate output filenames based on model path
    model_path_obj = Path(model_path)
    if model_path_obj.is_file():
        base_name = model_path_obj.stem
    else:
        # Sanitize model name for filename
        base_name = str(model_path).replace("/", "_").replace("\\", "_")
    
    confusion_matrix_path = str(output_dir / f"{base_name}_confusion_matrix_per_class.png")
    summary_path = str(output_dir / f"{base_name}_confusion_summary.png")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_confusion_matrices(
        predictions,
        labels,
        NEW_LABELS,
        output_path=confusion_matrix_path,
    )
    
    visualize_summary_matrix(
        predictions,
        labels,
        NEW_LABELS,
        output_path=summary_path,
    )
    
    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - Confusion matrices: {confusion_matrix_path}")
    print(f"  - Summary: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    typer.run(main)


