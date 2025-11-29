"""
Find optimal per-class classification thresholds for a BigEarthNet v2.0 model.

This script follows the recommended threshold tuning process:
1. Run model on Validation Set to get probabilities and labels
2. Find optimal threshold for each of the 19 classes individually
3. Save thresholds for later application to Test Set

Supports the same model input types as `visualize_confusion_matrix.py`:
1. HuggingFace pretrained models (model_name string)
2. Local BigEarthNet checkpoints (.ckpt file)
3. Local Multimodal checkpoints (.ckpt file)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer
from sklearn.metrics import f1_score

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import NEW_LABELS  # noqa: E402
from scripts.diagnose.model_utils import load_model_and_infer  # noqa: E402

__author__ = "Threshold diagnostics script (based on work by Leonard Hackel - BIFOLD/RSiM TU Berlin)"


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold_min: float = 0.01,
    threshold_max: float = 0.99,
    threshold_step: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find optimal threshold for each class individually.
    
    Args:
        y_true: [N, 19] Binary labels
        y_probs: [N, 19] Probabilities from model
        threshold_min: Minimum threshold to check
        threshold_max: Maximum threshold to check
        threshold_step: Step size for threshold sweep
    
    Returns:
        best_thresholds: [19] Optimal threshold for each class
        best_f1s: [19] Best F1 score for each class
    """
    n_classes = y_true.shape[1]
    best_thresholds = []
    best_f1s = []
    
    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
    
    print(f"\nFinding optimal thresholds for {n_classes} classes...")
    print(f"Checking thresholds from {threshold_min:.2f} to {threshold_max:.2f} (step={threshold_step:.2f})")
    
    # Iterate over each class (0 to 18)
    for i in range(n_classes):
        y_true_cls = y_true[:, i]
        y_prob_cls = y_probs[:, i]
        
        best_t = 0.5
        best_f1 = 0.0
        
        # Check thresholds
        for t in thresholds:
            preds = (y_prob_cls >= t).astype(int)
            f1 = f1_score(y_true_cls, preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        
        best_thresholds.append(best_t)
        best_f1s.append(best_f1)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{n_classes} classes...")
    
    return np.array(best_thresholds), np.array(best_f1s)


def _print_per_class_results(
    best_thresholds: np.ndarray,
    best_f1s: np.ndarray,
    class_names: list,
) -> None:
    """Print per-class threshold results in a formatted table."""
    print("\n" + "=" * 80)
    print("Per-Class Optimal Thresholds (found on Validation Set)")
    print("=" * 80)
    print(f"{'Class':<30} {'Threshold':<12} {'F1 Score':<10}")
    print("-" * 80)
    
    for i, (class_name, thr, f1) in enumerate(zip(class_names, best_thresholds, best_f1s)):
        print(f"{class_name:<30} {thr:<12.4f} {f1:<10.4f}")
    
    print("-" * 80)
    print(f"{'Average':<30} {np.mean(best_thresholds):<12.4f} {np.mean(best_f1s):<10.4f}")
    print("=" * 80)


def main(
    model_path: str = typer.Option(
        ...,
        help="Path to checkpoint file or HuggingFace model name "
        "(e.g., 'hackelle/resnet18-all-v0.1.1')",
    ),
    model_type: Optional[str] = typer.Option(
        None,
        help="Model type: 'hf_pretrained', 'checkpoint_bigearthnet', or "
        "'checkpoint_multimodal'. Auto-detected if not provided.",
    ),
    architecture: Optional[str] = typer.Option(
        None,
        help="Model architecture (auto-detected if not provided)",
    ),
    bandconfig: Optional[str] = typer.Option(
        None,
        help="Band configuration (auto-detected if not provided)",
    ),
    use_s1: Optional[bool] = typer.Option(
        None,
        "--use-s1/--no-use-s1",
        help="Whether to include S1 (for multimodal models)",
    ),
    seed: int = typer.Option(42, help="Random seed"),
    lr: float = typer.Option(0.001, help="Learning rate (for model initialization)"),
    drop_rate: float = typer.Option(0.15, help="Dropout rate"),
    drop_path_rate: float = typer.Option(0.15, help="Drop path rate"),
    warmup: int = typer.Option(1000, help="Warmup steps"),
    bs: int = typer.Option(32, help="Batch size"),
    workers: int = typer.Option(8, help="Number of workers"),
    threshold_min: float = typer.Option(
        0.01, help="Minimum threshold to sweep (inclusive, default: 0.01)"
    ),
    threshold_max: float = typer.Option(
        0.99, help="Maximum threshold to sweep (inclusive, default: 0.99)"
    ),
    threshold_step: float = typer.Option(
        0.01, help="Step size for threshold sweep (default: 0.01)"
    ),
    test_run: bool = typer.Option(
        False, help="Run with fewer batches for quick testing"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        help="Output directory for threshold file. If None, will save to same directory as model.",
    ),
    save_thresholds: bool = typer.Option(
        True,
        help="Save optimal thresholds to .npy file for later use on test set",
    ),
    dinov3_model_name: Optional[str] = typer.Option(
        None, help="DINOv3 HuggingFace model name"
    ),
):
    """
    Find optimal per-class thresholds on Validation Set for a BigEarthNet v2.0 model.
    
    This script:
    1. Runs model on Validation Set to collect probabilities and labels
    2. Finds optimal threshold for each of the 19 classes individually
    3. Saves thresholds to .npy file for application to Test Set
    
    The saved thresholds can then be applied to test set predictions:
        test_predictions = (test_probs >= optimal_thresholds).astype(int)
    """
    # Generate default output directory if not provided
    if output_dir is None:
        model_path_obj = Path(model_path)
        if model_path_obj.is_file():
            output_dir = str(model_path_obj.parent)
        else:
            output_dir = "."
        print(f"\nUsing default output directory: {output_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate threshold parameters
    if threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")
    if not (0.0 <= threshold_min < threshold_max <= 1.0):
        raise ValueError("threshold_min and threshold_max must be in [0, 1] with min < max")

    # Run inference on VALIDATION set to get probabilities and labels
    # We use threshold=0.5 just to get the probabilities; we'll optimize thresholds per-class
    print("\n" + "=" * 80)
    print("Step 1: Generating Validation Predictions")
    print("=" * 80)
    print("Running inference on VALIDATION set (threshold=0.5) to collect probabilities and labels...")
    _, probabilities, labels, _ = load_model_and_infer(
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
        threshold=0.5,
        split="validation",  # Use validation set for threshold tuning
    )

    if probabilities is None or labels is None:
        raise ValueError(
            "Failed to collect probabilities or labels. "
            "The validation set may be empty or the dataloader may not be configured correctly."
        )

    print(f"\nCollected {len(probabilities)} validation samples")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Labels shape: {labels.shape}")

    # Step 2: Find optimal thresholds per class
    print("\n" + "=" * 80)
    print("Step 2: Finding Optimal Thresholds per Class")
    print("=" * 80)
    best_thresholds, best_f1s = find_optimal_thresholds(
        y_true=labels,
        y_probs=probabilities,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )

    # Print results
    _print_per_class_results(best_thresholds, best_f1s, NEW_LABELS)

    # Calculate overall metrics with per-class thresholds
    per_class_preds = (probabilities >= best_thresholds).astype(int)
    overall_macro_f1 = f1_score(labels, per_class_preds, average="macro", zero_division=0)
    overall_micro_f1 = f1_score(labels, per_class_preds, average="micro", zero_division=0)

    print(f"\nOverall Performance with Per-Class Thresholds:")
    print(f"  Macro-averaged F1: {overall_macro_f1:.4f}")
    print(f"  Micro-averaged F1: {overall_micro_f1:.4f}")

    # Save thresholds to file
    if save_thresholds:
        model_path_obj = Path(model_path)
        if model_path_obj.is_file():
            base_name = model_path_obj.stem
        else:
            base_name = str(model_path).replace("/", "_").replace("\\", "_")
        
        threshold_file = output_dir / f"{base_name}_optimal_thresholds.npy"
        np.save(threshold_file, best_thresholds)
        print(f"\nSaved optimal thresholds to: {threshold_file}")
        print(f"  Shape: {best_thresholds.shape}")
        print(f"  You can load with: thresholds = np.load('{threshold_file}')")
        print(f"  Apply to test set: test_predictions = (test_probs >= thresholds).astype(int)")

    print("\n" + "=" * 80)
    print("Threshold Optimization Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Apply these thresholds to your TEST set (not validation!)")
    print("2. Use visualize_confusion_matrix.py with per-class thresholds if needed")
    print("3. Report final test metrics using the optimized thresholds")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    typer.run(main)


