"""
Compare two BigEarthNet v2.0 models (A/B test).

This script loads two models (HF pretrained, BigEarthNet checkpoint, or Multimodal checkpoint),
runs inference on the test set, and identifies which data samples perform better with model B compared to model A.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import NEW_LABELS  # noqa: E402
from scripts.diagnose.model_utils import load_model_and_infer  # noqa: E402

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def compute_sample_metrics(predictions: np.ndarray, labels: np.ndarray, probabilities: Optional[np.ndarray] = None) -> dict:
    """Compute per-sample F1, precision, recall, and AP metrics."""
    n_samples = len(predictions)
    metrics = {
        "f1": np.zeros(n_samples),
        "precision": np.zeros(n_samples),
        "recall": np.zeros(n_samples),
        "ap": np.zeros(n_samples),
    }

    for i in range(n_samples):
        y_true = labels[i]
        y_pred = predictions[i]

        if y_true.sum() == 0 and y_pred.sum() == 0:
            metrics["f1"][i] = metrics["precision"][i] = metrics["recall"][i] = metrics["ap"][i] = 1.0
            continue

        metrics["f1"][i] = f1_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["precision"][i] = precision_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["recall"][i] = recall_score(y_true, y_pred, average="micro", zero_division=0)

        if probabilities is not None:
            try:
                metrics["ap"][i] = average_precision_score(
                    y_true.reshape(1, -1),
                    probabilities[i].reshape(1, -1),
                    average="micro",
                )
            except Exception:
                metrics["ap"][i] = 0.0
        else:
            try:
                metrics["ap"][i] = average_precision_score(
                    y_true.reshape(1, -1),
                    y_pred.reshape(1, -1).astype(float),
                    average="micro",
                )
            except Exception:
                metrics["ap"][i] = 0.0

    return metrics


def compare_checkpoints(
    model_a_path: str,
    model_b_path: str,
    model_type_a: Optional[str] = None,
    model_type_b: Optional[str] = None,
    architecture_a: Optional[str] = None,
    architecture_b: Optional[str] = None,
    bandconfig_a: Optional[str] = None,
    bandconfig_b: Optional[str] = None,
    use_s1_a: Optional[bool] = None,
    use_s1_b: Optional[bool] = None,
    seed: int = 42,
    bs: int = 32,
    workers: int = 8,
    threshold: float = 0.5,
    test_run: bool = False,
    output_dir: str = "./compare_output",
    dinov3_model_name_a: Optional[str] = None,
    dinov3_model_name_b: Optional[str] = None,
):
    """Compare two models and identify where B performs better than A."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("COMPARING MODELS")
    print(f"{'=' * 60}")
    print(f"Model A: {model_a_path}")
    print(f"Model B: {model_b_path}")
    print(f"{'=' * 60}\n")

    preds_a, probs_a, labels, sample_ids = load_model_and_infer(
        model_path=model_a_path,
        model_type=model_type_a,
        architecture=architecture_a,
        bandconfig=bandconfig_a,
        use_s1=use_s1_a,
        seed=seed,
        bs=bs,
        workers=workers,
        test_run=test_run,
        dinov3_model_name=dinov3_model_name_a,
        threshold=threshold,
        allow_mock_data=False,
        return_sample_ids=True,
    )

    preds_b, probs_b, labels_b, _ = load_model_and_infer(
        model_path=model_b_path,
        model_type=model_type_b,
        architecture=architecture_b,
        bandconfig=bandconfig_b,
        use_s1=use_s1_b,
        seed=seed,
        bs=bs,
        workers=workers,
        test_run=test_run,
        dinov3_model_name=dinov3_model_name_b,
        threshold=threshold,
        allow_mock_data=False,
        return_sample_ids=True,
    )

    if len(preds_a) != len(preds_b):
        raise ValueError(
            f"Different number of samples: A has {len(preds_a)}, B has {len(preds_b)}. "
            "Make sure both models use the same test set configuration."
        )

    if not np.array_equal(labels, labels_b):
        print("Warning: Labels don't match between models. This might indicate different data ordering.")
        labels = labels_b

    print("\nComputing per-sample metrics...")
    metrics_a = compute_sample_metrics(preds_a, labels, probabilities=probs_a)
    metrics_b = compute_sample_metrics(preds_b, labels, probabilities=probs_b)

    differences = {
        "f1_diff": metrics_b["f1"] - metrics_a["f1"],
        "precision_diff": metrics_b["precision"] - metrics_a["precision"],
        "recall_diff": metrics_b["recall"] - metrics_a["recall"],
        "ap_diff": metrics_b["ap"] - metrics_a["ap"],
    }

    better_samples = {
        "f1": np.where(differences["f1_diff"] > 0)[0],
        "precision": np.where(differences["precision_diff"] > 0)[0],
        "recall": np.where(differences["recall_diff"] > 0)[0],
        "ap": np.where(differences["ap_diff"] > 0)[0],
    }

    print(f"\n{'=' * 60}")
    print("OVERALL STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(preds_a)}")
    print("\nSamples where B > A:")
    for metric_name in ["f1", "precision", "recall", "ap"]:
        count = len(better_samples[metric_name])
        print(f"  {metric_name.upper()}: {count} ({100 * count / len(preds_a):.1f}%)")

    print("\nAverage differences (B - A):")
    for key, value in differences.items():
        print(f"  {key.replace('_diff', '').upper()}: {value.mean():.4f}")
    print(f"{'=' * 60}\n")

    comparison_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "f1_a": metrics_a["f1"],
            "f1_b": metrics_b["f1"],
            "f1_diff": differences["f1_diff"],
            "precision_a": metrics_a["precision"],
            "precision_b": metrics_b["precision"],
            "precision_diff": differences["precision_diff"],
            "recall_a": metrics_a["recall"],
            "recall_b": metrics_b["recall"],
            "recall_diff": differences["recall_diff"],
            "ap_a": metrics_a["ap"],
            "ap_b": metrics_b["ap"],
            "ap_diff": differences["ap_diff"],
        }
    )

    for i, class_name in enumerate(NEW_LABELS):
        comparison_df[f"class_{i}_{class_name}_pred_a"] = preds_a[:, i]
        comparison_df[f"class_{i}_{class_name}_pred_b"] = preds_b[:, i]
        comparison_df[f"class_{i}_{class_name}_label"] = labels[:, i]
        comparison_df[f"class_{i}_{class_name}_prob_a"] = probs_a[:, i]
        comparison_df[f"class_{i}_{class_name}_prob_b"] = probs_b[:, i]

    csv_path = output_dir / "detailed_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved detailed comparison to: {csv_path}")

    better_samples_df = comparison_df[
        (comparison_df["f1_diff"] > 0)
        | (comparison_df["precision_diff"] > 0)
        | (comparison_df["recall_diff"] > 0)
        | (comparison_df["ap_diff"] > 0)
    ].copy()
    better_samples_df = better_samples_df.sort_values("f1_diff", ascending=False)
    better_csv_path = output_dir / "samples_where_b_better.csv"
    better_samples_df.to_csv(better_csv_path, index=False)
    print(f"Saved samples where B is better to: {better_csv_path}")
    print(f"  Found {len(better_samples_df)} samples where B performs better")

    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metric_names = ["f1_diff", "precision_diff", "recall_diff", "ap_diff"]
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        ax.hist(differences[metric], bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No difference")
        ax.axvline(differences[metric].mean(), color="green", linestyle="--", linewidth=2, label="Mean")
        ax.set_xlabel(f"{metric.replace('_diff', '').upper()} Difference (B - A)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {metric.replace('_diff', '').upper()} Differences")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    diff_fig = output_dir / "difference_distributions.png"
    plt.savefig(diff_fig, dpi=150, bbox_inches="tight")
    print(f"Saved difference distributions to: {diff_fig}")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    scatter_metrics = ["f1", "precision", "recall", "ap"]
    for i, metric in enumerate(scatter_metrics):
        ax = axes[i]
        ax.scatter(metrics_a[metric], metrics_b[metric], alpha=0.3, s=10)
        min_val = min(metrics_a[metric].min(), metrics_b[metric].min())
        max_val = max(metrics_a[metric].max(), metrics_b[metric].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y=x")
        ax.set_xlabel(f"Model A {metric.upper()}")
        ax.set_ylabel(f"Model B {metric.upper()}")
        ax.set_title(f"{metric.upper()}: A vs B")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_fig = output_dir / "scatter_comparison.png"
    plt.savefig(scatter_fig, dpi=150, bbox_inches="tight")
    print(f"Saved scatter comparison to: {scatter_fig}")
    plt.close()

    top_improvements = comparison_df.nlargest(20, "f1_diff")
    fig, ax = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(top_improvements))
    width = 0.35
    ax.bar(x_pos - width / 2, top_improvements["f1_a"], width, label="Model A", alpha=0.7)
    ax.bar(x_pos + width / 2, top_improvements["f1_b"], width, label="Model B", alpha=0.7)
    ax.set_xlabel("Sample ID (Top 20 Improvements)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Top 20 Samples with Largest F1 Improvement (B vs A)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_improvements["sample_id"], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    improvements_fig = output_dir / "top_improvements.png"
    plt.savefig(improvements_fig, dpi=150, bbox_inches="tight")
    print(f"Saved top improvements to: {improvements_fig}")
    plt.close()

    print(f"\n{'=' * 60}")
    print("Comparison complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}\n")

    return comparison_df, better_samples_df


def main(
    model_a: str = typer.Option(..., help="Path to model A (checkpoint file or HuggingFace model name)"),
    model_b: str = typer.Option(..., help="Path to model B (checkpoint file or HuggingFace model name)"),
    model_type_a: Optional[str] = typer.Option(None, help="Model type for A (auto-detected if not provided)."),
    model_type_b: Optional[str] = typer.Option(None, help="Model type for B (auto-detected if not provided)."),
    architecture_a: Optional[str] = typer.Option(None, help="Architecture for model A (auto-detected if not provided)"),
    architecture_b: Optional[str] = typer.Option(None, help="Architecture for model B (auto-detected if not provided)"),
    bandconfig_a: Optional[str] = typer.Option(None, help="Band config for model A (auto-detected if not provided)"),
    bandconfig_b: Optional[str] = typer.Option(None, help="Band config for model B (auto-detected if not provided)"),
    use_s1_a: Optional[bool] = typer.Option(None, "--use-s1-a/--no-use-s1-a", help="Whether to include S1 for model A"),
    use_s1_b: Optional[bool] = typer.Option(None, "--use-s1-b/--no-use-s1-b", help="Whether to include S1 for model B"),
    seed: int = typer.Option(42, help="Random seed"),
    bs: int = typer.Option(32, help="Batch size"),
    workers: int = typer.Option(8, help="Number of workers"),
    threshold: float = typer.Option(0.5, help="Classification threshold"),
    test_run: bool = typer.Option(False, help="Run with fewer batches for quick testing"),
    output_dir: str = typer.Option("./compare_output", help="Output directory for results"),
    dinov3_model_name_a: Optional[str] = typer.Option(None, help="DINOv3 HuggingFace model name for model A"),
    dinov3_model_name_b: Optional[str] = typer.Option(None, help="DINOv3 HuggingFace model name for model B"),
):
    compare_checkpoints(
        model_a_path=model_a,
        model_b_path=model_b,
        model_type_a=model_type_a,
        model_type_b=model_type_b,
        architecture_a=architecture_a,
        architecture_b=architecture_b,
        bandconfig_a=bandconfig_a,
        bandconfig_b=bandconfig_b,
        use_s1_a=use_s1_a,
        use_s1_b=use_s1_b,
        seed=seed,
        bs=bs,
        workers=workers,
        threshold=threshold,
        test_run=test_run,
        output_dir=output_dir,
        dinov3_model_name_a=dinov3_model_name_a,
        dinov3_model_name_b=dinov3_model_name_b,
    )


if __name__ == "__main__":
    typer.run(main)
"""
Compare two BigEarthNet v2.0 models (A/B test).

This script loads two models (HF pretrained, BigEarthNet checkpoint, or Multimodal checkpoint),
runs inference on the test set, and identifies which data samples perform better with model B.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import NEW_LABELS  # noqa: E402
from diagnose.model_utils import load_model_and_infer  # noqa: E402

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def compute_sample_metrics(predictions: np.ndarray, labels: np.ndarray, probabilities: Optional[np.ndarray] = None) -> dict:
    """Compute per-sample F1, precision, recall, and AP metrics."""
    n_samples = len(predictions)
    metrics = {
        "f1": np.zeros(n_samples),
        "precision": np.zeros(n_samples),
        "recall": np.zeros(n_samples),
        "ap": np.zeros(n_samples),
    }

    for i in range(n_samples):
        y_true = labels[i]
        y_pred = predictions[i]

        if y_true.sum() == 0 and y_pred.sum() == 0:
            metrics["f1"][i] = metrics["precision"][i] = metrics["recall"][i] = metrics["ap"][i] = 1.0
            continue

        metrics["f1"][i] = f1_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["precision"][i] = precision_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["recall"][i] = recall_score(y_true, y_pred, average="micro", zero_division=0)

        if probabilities is not None:
            try:
                metrics["ap"][i] = average_precision_score(
                    y_true.reshape(1, -1),
                    probabilities[i].reshape(1, -1),
                    average="micro",
                )
            except Exception:
                metrics["ap"][i] = 0.0
        else:
            try:
                metrics["ap"][i] = average_precision_score(
                    y_true.reshape(1, -1),
                    y_pred.reshape(1, -1).astype(float),
                    average="micro",
                )
            except Exception:
                metrics["ap"][i] = 0.0

    return metrics


def compare_checkpoints(
    model_a_path: str,
    model_b_path: str,
    model_type_a: Optional[str] = None,
    model_type_b: Optional[str] = None,
    architecture_a: Optional[str] = None,
    architecture_b: Optional[str] = None,
    bandconfig_a: Optional[str] = None,
    bandconfig_b: Optional[str] = None,
    use_s1_a: Optional[bool] = None,
    use_s1_b: Optional[bool] = None,
    seed: int = 42,
    bs: int = 32,
    workers: int = 8,
    threshold: float = 0.5,
    test_run: bool = False,
    output_dir: str = "./compare_output",
    dinov3_model_name_a: Optional[str] = None,
    dinov3_model_name_b: Optional[str] = None,
):
    """Compare two models and identify where B performs better than A."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("COMPARING MODELS")
    print(f"{'=' * 60}")
    print(f"Model A: {model_a_path}")
    print(f"Model B: {model_b_path}")
    print(f"{'=' * 60}\n")

    preds_a, probs_a, labels, sample_ids = load_model_and_infer(
        model_path=model_a_path,
        model_type=model_type_a,
        architecture=architecture_a,
        bandconfig=bandconfig_a,
        use_s1=use_s1_a,
        seed=seed,
        bs=bs,
        workers=workers,
        test_run=test_run,
        dinov3_model_name=dinov3_model_name_a,
        threshold=threshold,
        allow_mock_data=False,
        return_sample_ids=True,
    )

    preds_b, probs_b, labels_b, _ = load_model_and_infer(
        model_path=model_b_path,
        model_type=model_type_b,
        architecture=architecture_b,
        bandconfig=bandconfig_b,
        use_s1=use_s1_b,
        seed=seed,
        bs=bs,
        workers=workers,
        test_run=test_run,
        dinov3_model_name=dinov3_model_name_b,
        threshold=threshold,
        allow_mock_data=False,
        return_sample_ids=True,
    )

    if len(preds_a) != len(preds_b):
        raise ValueError(
            f"Different number of samples: A has {len(preds_a)}, B has {len(preds_b)}. "
            "Make sure both models use the same test set configuration."
        )

    if not np.array_equal(labels, labels_b):
        print("Warning: Labels don't match between models. This might indicate different data ordering.")
        labels = labels_b

    print("\nComputing per-sample metrics...")
    metrics_a = compute_sample_metrics(preds_a, labels, probabilities=probs_a)
    metrics_b = compute_sample_metrics(preds_b, labels, probabilities=probs_b)

    differences = {
        "f1_diff": metrics_b["f1"] - metrics_a["f1"],
        "precision_diff": metrics_b["precision"] - metrics_a["precision"],
        "recall_diff": metrics_b["recall"] - metrics_a["recall"],
        "ap_diff": metrics_b["ap"] - metrics_a["ap"],
    }

    better_samples = {
        "f1": np.where(differences["f1_diff"] > 0)[0],
        "precision": np.where(differences["precision_diff"] > 0)[0],
        "recall": np.where(differences["recall_diff"] > 0)[0],
        "ap": np.where(differences["ap_diff"] > 0)[0],
    }

    print(f"\n{'=' * 60}")
    print("OVERALL STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(preds_a)}")
    print("\nSamples where B > A:")
    for metric_name in ["f1", "precision", "recall", "ap"]:
        count = len(better_samples[metric_name])
        print(f"  {metric_name.upper()}: {count} ({100 * count / len(preds_a):.1f}%)")

    print("\nAverage differences (B - A):")
    for key, value in differences.items():
        print(f"  {key.replace('_diff', '').upper()}: {value.mean():.4f}")
    print(f"{'=' * 60}\n")

    comparison_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "f1_a": metrics_a["f1"],
            "f1_b": metrics_b["f1"],
            "f1_diff": differences["f1_diff"],
            "precision_a": metrics_a["precision"],
            "precision_b": metrics_b["precision"],
            "precision_diff": differences["precision_diff"],
            "recall_a": metrics_a["recall"],
            "recall_b": metrics_b["recall"],
            "recall_diff": differences["recall_diff"],
            "ap_a": metrics_a["ap"],
            "ap_b": metrics_b["ap"],
            "ap_diff": differences["ap_diff"],
        }
    )

    for i, class_name in enumerate(NEW_LABELS):
        comparison_df[f"class_{i}_{class_name}_pred_a"] = preds_a[:, i]
        comparison_df[f"class_{i}_{class_name}_pred_b"] = preds_b[:, i]
        comparison_df[f"class_{i}_{class_name}_label"] = labels[:, i]
        comparison_df[f"class_{i}_{class_name}_prob_a"] = probs_a[:, i]
        comparison_df[f"class_{i}_{class_name}_prob_b"] = probs_b[:, i]

    csv_path = output_dir / "detailed_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved detailed comparison to: {csv_path}")

    better_samples_df = comparison_df[
        (comparison_df["f1_diff"] > 0)
        | (comparison_df["precision_diff"] > 0)
        | (comparison_df["recall_diff"] > 0)
        | (comparison_df["ap_diff"] > 0)
    ].copy()
    better_samples_df = better_samples_df.sort_values("f1_diff", ascending=False)
    better_csv_path = output_dir / "samples_where_b_better.csv"
    better_samples_df.to_csv(better_csv_path, index=False)
    print(f"Saved samples where B is better to: {better_csv_path}")
    print(f"  Found {len(better_samples_df)} samples where B performs better")

    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    metric_names = ["f1_diff", "precision_diff", "recall_diff", "ap_diff"]
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        ax.hist(differences[metric], bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No difference")
        ax.axvline(differences[metric].mean(), color="green", linestyle="--", linewidth=2, label="Mean")
        ax.set_xlabel(f"{metric.replace('_diff', '').upper()} Difference (B - A)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {metric.replace('_diff', '').upper()} Differences")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    diff_fig = output_dir / "difference_distributions.png"
    plt.savefig(diff_fig, dpi=150, bbox_inches="tight")
    print(f"Saved difference distributions to: {diff_fig}")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    scatter_metrics = ["f1", "precision", "recall", "ap"]
    for i, metric in enumerate(scatter_metrics):
        ax = axes[i]
        ax.scatter(metrics_a[metric], metrics_b[metric], alpha=0.3, s=10)
        min_val = min(metrics_a[metric].min(), metrics_b[metric].min())
        max_val = max(metrics_a[metric].max(), metrics_b[metric].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y=x")
        ax.set_xlabel(f"Model A {metric.upper()}")
        ax.set_ylabel(f"Model B {metric.UPPER()}")

"""
Compare two BigEarthNet v2.0 models (A/B test).

This script loads two models (HF pretrained, BigEarthNet checkpoint, or Multimodal checkpoint),
runs inference on the test set, and identifies which data samples perform better with model B compared to model A.

Supports three model types:
1. HuggingFace pretrained models (model_name string)
2. Local BigEarthNet checkpoints (.ckpt file)
3. Local Multimodal checkpoints (.ckpt file)
"""
import sys
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path to allow importing reben_publication
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import lightning.pytorch as pl
import torch
import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BENv2_utils import resolve_data_dir, NEW_LABELS, STANDARD_BANDS
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
from huggingface_hub import HfApi

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from multimodal.lightning_module import MultiModalLightningModule
from scripts.utils import get_benv2_dir_dict, get_bands, default_dm

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def _infer_dinov3_model_from_checkpoint(checkpoint_path: str) -> str:
    """Infer DINOv3 model name from checkpoint."""
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
    # Check if it's a HuggingFace model name (contains '/' and doesn't end with .ckpt)
    if '/' in model_path and not model_path.endswith('.ckpt') and not Path(model_path).exists():
        return 'hf_pretrained'
    
    # Check if it's a file path
    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        return 'hf_pretrained'
    
    # Try to load checkpoint and detect type
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', {})
        hparams = checkpoint.get('hyper_parameters', {})
        
        multimodal_keys = ['model.dinov3_backbone', 'model.resnet_backbone', 'model.fusion']
        has_multimodal_keys = any(key in str(state_dict.keys()) for key in multimodal_keys)
        
        if 'use_s1' in hparams or 'fusion_type' in hparams:
            return 'checkpoint_multimodal'
        
        if has_multimodal_keys:
            return 'checkpoint_multimodal'
        
        return 'checkpoint_bigearthnet'
    except:
        return 'checkpoint_bigearthnet'


def load_model_and_infer(
    model_path: str,
    model_type: Optional[str] = None,
    architecture: Optional[str] = None,
    bandconfig: Optional[str] = None,
    use_s1: Optional[bool] = None,
    seed: int = 42,
    lr: float = 0.001,
    drop_rate: float = 0.15,
    drop_path_rate: float = 0.15,
    warmup: int = 1000,
    bs: int = 32,
    workers: int = 8,
    test_run: bool = False,
    dinov3_model_name: Optional[str] = None,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load model (HF pretrained, BigEarthNet checkpoint, or Multimodal checkpoint) and run inference.
    
    Returns:
        predictions: Binary predictions (n_samples, n_classes)
        probabilities: Prediction probabilities (n_samples, n_classes)
        labels: True labels (n_samples, n_classes)
        sample_ids: Sample indices (n_samples,)
    """
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory."
    
    # Detect model type if not provided
    if model_type is None:
        model_type = _detect_model_type(model_path)
    
    print(f"\n{'='*60}")
    print(f"Model Type: {model_type}")
    print(f"Model Path: {model_path}")
    print(f"{'='*60}\n")
    
    num_classes = 19
    img_size = 120
    
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    # ============================================================================
    # LOAD HUGGINGFACE PRETRAINED MODEL
    # ============================================================================
    if model_type == 'hf_pretrained':
        print("Loading HuggingFace pretrained model...")
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_path)
        model.eval()
        
        channels = model.config.channels
        architecture = getattr(model.config, 'timm_model_name', 'unknown')
        
        channel_to_bandconfig = {
            12: "all", 10: "s2", 2: "s1", 13: "all_full",
            11: "s2_full", 3: "rgb"
        }
        bandconfig = channel_to_bandconfig.get(channels, "s2")
        
        hparams = {
            "architecture": architecture,
            "seed": seed,
            "lr": lr,
            "epochs": 0,
            "batch_size": bs,
            "workers": workers,
            "channels": channels,
            "dropout": drop_rate,
            "drop_path_rate": drop_path_rate,
            "bandconfig": bandconfig,
            "warmup": warmup,
        }
        
        hostname, data_dirs = get_benv2_dir_dict()
        data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
        dm = default_dm(hparams, data_dirs, img_size)
    
    # ============================================================================
    # LOAD BIGEARTHNET CHECKPOINT
    # ============================================================================
    elif model_type == 'checkpoint_bigearthnet':
        print("Loading BigEarthNet checkpoint...")
        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        checkpoint_channels = None
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'hyper_parameters' in checkpoint:
            hparams_ckpt = checkpoint['hyper_parameters']
            if architecture is None and 'architecture' in hparams_ckpt:
                architecture = hparams_ckpt['architecture']
            if bandconfig is None and 'bandconfig' in hparams_ckpt:
                bandconfig = hparams_ckpt['bandconfig']
            if 'channels' in hparams_ckpt:
                checkpoint_channels = hparams_ckpt['channels']
            if 'dropout' in hparams_ckpt and drop_rate == 0.15:
                drop_rate = hparams_ckpt['dropout']
            if 'drop_path_rate' in hparams_ckpt and drop_path_rate == 0.15:
                drop_path_rate = hparams_ckpt['drop_path_rate']
            if 'seed' in hparams_ckpt and seed == 42:
                seed = hparams_ckpt['seed']
        
        if architecture is None or bandconfig is None:
            filename_parts = ckpt_path.stem.split('-')
            if len(filename_parts) >= 3 and architecture is None:
                potential_arch = filename_parts[0]
                if potential_arch and not potential_arch.isdigit():
                    architecture = potential_arch
            
            if bandconfig is None:
                channels = checkpoint_channels
                if channels is None:
                    try:
                        channels = int(filename_parts[2])
                    except (ValueError, IndexError):
                        channels = None
                
                if channels is not None:
                    channel_to_bandconfig = {
                        12: "all", 10: "s2", 2: "s1", 13: "all_full",
                        11: "s2_full", 3: "rgb"
                    }
                    bandconfig = channel_to_bandconfig.get(channels, "s2")
        
        if architecture is None:
            raise ValueError("Could not infer architecture. Please provide --architecture explicitly.")
        if bandconfig is None:
            raise ValueError("Could not infer bandconfig. Please provide --bandconfig explicitly.")
        
        bands, channels = get_bands(bandconfig)
        
        if checkpoint_channels is not None and channels != checkpoint_channels:
            raise ValueError(
                f"Channel mismatch! Checkpoint has {checkpoint_channels} channels, "
                f"but bandconfig '{bandconfig}' produces {channels} channels."
            )
        
        config = ILMConfiguration(
            network_type=ILMType.IMAGE_CLASSIFICATION,
            classes=num_classes,
            image_size=img_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            timm_model_name=architecture,
            channels=channels,
        )
        
        dinov3_name = dinov3_model_name
        if architecture.startswith('dinov3') and dinov3_name is None:
            if 'small' in architecture.lower() or 's' in architecture.lower():
                dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
            elif 'base' in architecture.lower() or 'b' in architecture.lower():
                dinov3_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            elif 'large' in architecture.lower() or 'l' in architecture.lower():
                dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            elif 'giant' in architecture.lower() or 'g' in architecture.lower():
                dinov3_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
            else:
                dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        
        model = BigEarthNetv2_0_ImageClassifier(config, lr=lr, warmup=warmup, dinov3_model_name=dinov3_name)
        
        hparams = {
            "architecture": architecture,
            "seed": seed,
            "lr": lr,
            "epochs": 0,
            "batch_size": bs,
            "workers": workers,
            "channels": channels,
            "dropout": drop_rate,
            "drop_path_rate": drop_path_rate,
            "bandconfig": bandconfig,
            "warmup": warmup,
        }
        
        hostname, data_dirs = get_benv2_dir_dict()
        data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
        dm = default_dm(hparams, data_dirs, img_size)
        
        trainer = pl.Trainer(accelerator="auto", logger=False)
        model = model.load_from_checkpoint(str(ckpt_path))
        model.eval()
    
    # ============================================================================
    # LOAD MULTIMODAL CHECKPOINT
    # ============================================================================
    elif model_type == 'checkpoint_multimodal':
        print("Loading Multimodal checkpoint...")
        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        
        checkpoint_channels = None
        checkpoint_use_s1 = None
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'hyper_parameters' in checkpoint:
            hparams_ckpt = checkpoint['hyper_parameters']
            if architecture is None and 'architecture' in hparams_ckpt:
                architecture = hparams_ckpt['architecture']
            if bandconfig is None and 'bandconfig' in hparams_ckpt:
                bandconfig = hparams_ckpt['bandconfig']
            if use_s1 is None and 'use_s1' in hparams_ckpt:
                checkpoint_use_s1 = hparams_ckpt['use_s1']
                use_s1 = checkpoint_use_s1
            if 'channels' in hparams_ckpt:
                checkpoint_channels = hparams_ckpt['channels']
            if 'dropout' in hparams_ckpt and drop_rate == 0.15:
                drop_rate = hparams_ckpt['dropout']
            if 'seed' in hparams_ckpt and seed == 42:
                seed = hparams_ckpt['seed']
        
        if dinov3_model_name is None:
            if architecture is not None:
                arch_lower = architecture.lower().replace("_", "-")
                if "giant" in arch_lower or arch_lower.endswith("-g"):
                    dinov3_model_name = "facebook/dinov3-vitg16-pretrain-lvd1689m"
                elif "large" in arch_lower or arch_lower.endswith("-l"):
                    dinov3_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
                elif "base" in arch_lower or arch_lower.endswith("-b"):
                    dinov3_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
                elif "small" in arch_lower or arch_lower.endswith("-s"):
                    dinov3_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
                else:
                    dinov3_model_name = _infer_dinov3_model_from_checkpoint(model_path)
            else:
                dinov3_model_name = _infer_dinov3_model_from_checkpoint(model_path)
        
        if bandconfig is None:
            if checkpoint_channels is not None:
                if checkpoint_channels == 3:
                    bandconfig = "rgb"
                elif checkpoint_channels == 12:
                    bandconfig = "s2"
                elif checkpoint_channels == 14:
                    bandconfig = "s2s1"
                else:
                    bandconfig = "s2s1" if checkpoint_channels > 12 else "s2"
            else:
                bandconfig = "s2s1"
        
        band_alias = bandconfig.lower()
        band_map = {
            "rgb": "rgb", "truecolor": "rgb",
            "s2": "s2", "s2_only": "s2",
            "s2s1": "s2s1", "all": "s2s1", "multimodal": "s2s1",
        }
        effective_bandconfig = band_map.get(band_alias, "s2s1")
        
        if use_s1 is None:
            use_s1 = (effective_bandconfig == "s2s1")
            if checkpoint_use_s1 is not None:
                use_s1 = checkpoint_use_s1
        
        s2_bands = STANDARD_BANDS.get("S2", STANDARD_BANDS.get("s2_full", []))
        s1_bands = STANDARD_BANDS.get("S1", ["VV", "VH"])
        rgb_bands = ["B04", "B03", "B02"]
        full_s2_non_rgb = [b for b in s2_bands if b not in rgb_bands]
        
        if effective_bandconfig == "rgb":
            s2_non_rgb = []
        else:
            s2_non_rgb = full_s2_non_rgb
        
        s2_ordered = rgb_bands + s2_non_rgb
        resnet_input_channels = len(s2_non_rgb) + (len(s1_bands) if use_s1 else 0)
        use_resnet = resnet_input_channels > 0
        
        if effective_bandconfig == "rgb":
            multimodal_bands = rgb_bands
        elif use_s1:
            multimodal_bands = s2_ordered + s1_bands
        else:
            multimodal_bands = s2_ordered
        
        num_channels = len(multimodal_bands)
        
        STANDARD_BANDS[num_channels] = multimodal_bands
        STANDARD_BANDS["multimodal"] = multimodal_bands
        BENv2DataSet.channel_configurations[num_channels] = multimodal_bands
        BENv2DataSet.avail_chan_configs[num_channels] = "Multimodal"
        
        config = {
            "backbones": {
                "dinov3": {
                    "model_name": dinov3_model_name,
                    "pretrained": True,
                    "freeze": False,
                    "lr": 1e-4,
                },
                "resnet101": {
                    "input_channels": resnet_input_channels,
                    "pretrained": True,
                    "freeze": False,
                    "lr": 1e-4,
                    "enabled": use_resnet,
                },
            },
            "fusion": {"type": "concat"},
            "classifier": {
                "type": "linear",
                "num_classes": num_classes,
                "drop_rate": drop_rate,
            },
            "image_size": img_size,
        }
        
        try:
            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                if 'fusion_type' in hparams:
                    config["fusion"]["type"] = hparams['fusion_type']
                if 'classifier_type' in hparams:
                    config["classifier"]["type"] = hparams['classifier_type']
                if 'classifier_hidden_dim' in hparams and config["classifier"]["type"] == "mlp":
                    config["classifier"]["hidden_dim"] = hparams['classifier_hidden_dim']
        except:
            pass
        
        model = MultiModalLightningModule(
            config=config,
            lr=lr,
            warmup=None if warmup == -1 else warmup,
            dinov3_checkpoint=None,
            resnet_checkpoint=None,
            freeze_dinov3=False,
            freeze_resnet=False,
            dinov3_model_name=dinov3_model_name,
            threshold=threshold,
        )
        
        hparams = {
            "architecture": architecture or "multimodal",
            "seed": seed,
            "lr": lr,
            "epochs": 0,
            "batch_size": bs,
            "workers": workers,
            "channels": num_channels,
            "dropout": drop_rate,
            "bandconfig": effective_bandconfig,
            "warmup": warmup,
            "use_s1": use_s1,
        }
        
        hostname, data_dirs = get_benv2_dir_dict()
        data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
        dm = default_dm(hparams, data_dirs, img_size)
        
        trainer = pl.Trainer(accelerator="auto", logger=False)
        model = model.load_from_checkpoint(str(ckpt_path))
        model.eval()
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Architecture: {architecture or 'unknown'}")
    print(f"Band config: {bandconfig or 'unknown'}")
    print(f"Channels: {hparams.get('channels', 'unknown')}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}\n")
    
    # Run inference
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    test_loader = dm.test_dataloader()
    device = next(model.parameters()).device
    
    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            
            all_predictions.append(preds.cpu())
            all_probabilities.append(probs.cpu())
            all_labels.append(y.cpu())
            
            if test_run and batch_idx >= 4:
                break
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
    
    predictions = torch.cat(all_predictions, dim=0).numpy()
    probabilities = torch.cat(all_probabilities, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    sample_ids = np.arange(len(predictions))
    
    print(f"Collected predictions for {len(predictions)} samples")
    
    return predictions, probabilities, labels, sample_ids


def compute_sample_metrics(predictions, labels, probabilities=None):
    """
    Compute per-sample metrics (F1, precision, recall, AP).
    
    Args:
        predictions: Binary predictions (n_samples, n_classes)
        labels: True labels (n_samples, n_classes)
        probabilities: Prediction probabilities (n_samples, n_classes), optional
    
    Returns:
        Dictionary with metrics per sample
    """
    n_samples = len(predictions)
    metrics = {
        'f1': np.zeros(n_samples),
        'precision': np.zeros(n_samples),
        'recall': np.zeros(n_samples),
        'ap': np.zeros(n_samples),
    }
    
    for i in range(n_samples):
        y_true = labels[i]
        y_pred = predictions[i]
        
        # Skip if no positive labels (both true and pred are all zeros)
        if y_true.sum() == 0 and y_pred.sum() == 0:
            metrics['f1'][i] = 1.0
            metrics['precision'][i] = 1.0
            metrics['recall'][i] = 1.0
            metrics['ap'][i] = 1.0
            continue
        
        # Compute metrics
        metrics['f1'][i] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['precision'][i] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall'][i] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Average precision (use probabilities if available, otherwise use binary predictions)
        if probabilities is not None:
            try:
                metrics['ap'][i] = average_precision_score(
                    y_true.reshape(1, -1),
                    probabilities[i].reshape(1, -1),
                    average='micro'
                )
            except:
                metrics['ap'][i] = 0.0
        else:
            # Fallback: approximate with binary predictions
            try:
                metrics['ap'][i] = average_precision_score(
                    y_true.reshape(1, -1),
                    y_pred.reshape(1, -1).astype(float),
                    average='micro'
                )
            except:
                metrics['ap'][i] = 0.0
    
    return metrics


def compare_checkpoints(
    model_a_path: str,
    model_b_path: str,
    model_type_a: Optional[str] = None,
    model_type_b: Optional[str] = None,
    architecture_a: Optional[str] = None,
    architecture_b: Optional[str] = None,
    bandconfig_a: Optional[str] = None,
    bandconfig_b: Optional[str] = None,
    use_s1_a: Optional[bool] = None,
    use_s1_b: Optional[bool] = None,
    seed: int = 42,
    bs: int = 32,
    workers: int = 8,
    threshold: float = 0.5,
    test_run: bool = False,
    output_dir: str = "./compare_output",
    dinov3_model_name_a: Optional[str] = None,
    dinov3_model_name_b: Optional[str] = None,
):
    """
    Compare two models and identify where B performs better than A.
    
    Supports three model types for each model:
    1. HuggingFace pretrained: Provide model name (e.g., 'hackelle/resnet18-all-v0.1.1')
    2. BigEarthNet checkpoint: Provide path to .ckpt file
    3. Multimodal checkpoint: Provide path to .ckpt file (auto-detected)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("COMPARING MODELS")
    print(f"{'='*60}")
    print(f"Model A: {model_a_path}")
    print(f"Model B: {model_b_path}")
    print(f"{'='*60}\n")
    
    # Load and run inference for model A
    print("Loading model A...")
    preds_a, probs_a, labels, sample_ids = load_model_and_infer(
        model_path=model_a_path,
        model_type=model_type_a,
        architecture=architecture_a,
        bandconfig=bandconfig_a,
        use_s1=use_s1_a,
        seed=seed,
        bs=bs,
        workers=workers,
        test_run=test_run,
        threshold=threshold,
        dinov3_model_name=dinov3_model_name_a,
    )
    
    # Load and run inference for model B
    print("\nLoading model B...")
    preds_b, probs_b, labels_b, sample_ids_b = load_model_and_infer(
        model_path=model_b_path,
        model_type=model_type_b,
        architecture=architecture_b,
        bandconfig=bandconfig_b,
        use_s1=use_s1_b,
        seed=seed,
        bs=bs,
        workers=workers,
        test_run=test_run,
        threshold=threshold,
        dinov3_model_name=dinov3_model_name_b,
    )
    
    # Verify same number of samples
    if len(preds_a) != len(preds_b):
        raise ValueError(
            f"Different number of samples: A has {len(preds_a)}, B has {len(preds_b)}. "
            "Make sure both checkpoints use the same test set configuration."
        )
    
    # Verify labels match (to ensure we're comparing the same samples)
    if not np.array_equal(labels, labels_b):
        print("Warning: Labels don't match between checkpoints. This might indicate different data ordering.")
        print("Attempting to align samples...")
        # Try to find matching samples (this is a fallback, ideally they should match)
        # For now, we'll proceed but warn the user
        labels = labels_b  # Use labels from checkpoint B
    
    # Compute per-sample metrics for both checkpoints
    print("\nComputing per-sample metrics...")
    metrics_a = compute_sample_metrics(preds_a, labels, probabilities=probs_a)
    metrics_b = compute_sample_metrics(preds_b, labels, probabilities=probs_b)
    
    # Compute differences (B - A)
    differences = {
        'f1_diff': metrics_b['f1'] - metrics_a['f1'],
        'precision_diff': metrics_b['precision'] - metrics_a['precision'],
        'recall_diff': metrics_b['recall'] - metrics_a['recall'],
        'ap_diff': metrics_b['ap'] - metrics_a['ap'],
    }
    
    # Identify samples where B is better than A
    better_samples = {
        'f1': np.where(differences['f1_diff'] > 0)[0],
        'precision': np.where(differences['precision_diff'] > 0)[0],
        'recall': np.where(differences['recall_diff'] > 0)[0],
        'ap': np.where(differences['ap_diff'] > 0)[0],
    }
    
    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"Total samples: {len(preds_a)}")
    print(f"\nSamples where B > A:")
    print(f"  F1: {len(better_samples['f1'])} ({100*len(better_samples['f1'])/len(preds_a):.1f}%)")
    print(f"  Precision: {len(better_samples['precision'])} ({100*len(better_samples['precision'])/len(preds_a):.1f}%)")
    print(f"  Recall: {len(better_samples['recall'])} ({100*len(better_samples['recall'])/len(preds_a):.1f}%)")
    print(f"  AP: {len(better_samples['ap'])} ({100*len(better_samples['ap'])/len(preds_a):.1f}%)")
    
    # Average differences
    print(f"\nAverage differences (B - A):")
    print(f"  F1: {differences['f1_diff'].mean():.4f}")
    print(f"  Precision: {differences['precision_diff'].mean():.4f}")
    print(f"  Recall: {differences['recall_diff'].mean():.4f}")
    print(f"  AP: {differences['ap_diff'].mean():.4f}")
    print(f"{'='*60}\n")
    
    # Create detailed comparison DataFrame
    comparison_df = pd.DataFrame({
        'sample_id': sample_ids,
        'f1_a': metrics_a['f1'],
        'f1_b': metrics_b['f1'],
        'f1_diff': differences['f1_diff'],
        'precision_a': metrics_a['precision'],
        'precision_b': metrics_b['precision'],
        'precision_diff': differences['precision_diff'],
        'recall_a': metrics_a['recall'],
        'recall_b': metrics_b['recall'],
        'recall_diff': differences['recall_diff'],
        'ap_a': metrics_a['ap'],
        'ap_b': metrics_b['ap'],
        'ap_diff': differences['ap_diff'],
    })
    
    # Add class-level information
    for i, class_name in enumerate(NEW_LABELS):
        comparison_df[f'class_{i}_{class_name}_pred_a'] = preds_a[:, i]
        comparison_df[f'class_{i}_{class_name}_pred_b'] = preds_b[:, i]
        comparison_df[f'class_{i}_{class_name}_label'] = labels[:, i]
        comparison_df[f'class_{i}_{class_name}_prob_a'] = probs_a[:, i]
        comparison_df[f'class_{i}_{class_name}_prob_b'] = probs_b[:, i]
    
    # Save detailed comparison
    csv_path = output_dir / "detailed_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved detailed comparison to: {csv_path}")
    
    # Save samples where B is better
    better_samples_df = comparison_df[
        (comparison_df['f1_diff'] > 0) | 
        (comparison_df['precision_diff'] > 0) | 
        (comparison_df['recall_diff'] > 0) | 
        (comparison_df['ap_diff'] > 0)
    ].copy()
    better_samples_df = better_samples_df.sort_values('f1_diff', ascending=False)
    
    better_csv_path = output_dir / "samples_where_b_better.csv"
    better_samples_df.to_csv(better_csv_path, index=False)
    print(f"Saved samples where B is better to: {better_csv_path}")
    print(f"  Found {len(better_samples_df)} samples where B performs better")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Distribution of differences
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['f1_diff', 'precision_diff', 'recall_diff', 'ap_diff']
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        ax.hist(differences[metric], bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
        ax.axvline(differences[metric].mean(), color='green', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel(f'{metric.replace("_diff", "").upper()} Difference (B - A)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {metric.replace("_diff", "").upper()} Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "difference_distributions.png", dpi=150, bbox_inches='tight')
    print(f"Saved difference distributions to: {output_dir / 'difference_distributions.png'}")
    plt.close()
    
    # 2. Scatter plots: A vs B
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_to_scatter = ['f1', 'precision', 'recall', 'ap']
    for i, metric in enumerate(metrics_to_scatter):
        ax = axes[i]
        ax.scatter(metrics_a[metric], metrics_b[metric], alpha=0.3, s=10)
        
        # Add diagonal line
        min_val = min(metrics_a[metric].min(), metrics_b[metric].min())
        max_val = max(metrics_a[metric].max(), metrics_b[metric].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        
        ax.set_xlabel(f'Checkpoint A {metric.upper()}')
        ax.set_ylabel(f'Checkpoint B {metric.upper()}')
        ax.set_title(f'{metric.upper()}: A vs B')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved scatter comparison to: {output_dir / 'scatter_comparison.png'}")
    plt.close()
    
    # 3. Top improvements
    top_improvements = comparison_df.nlargest(20, 'f1_diff')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(top_improvements))
    width = 0.35
    
    ax.bar(x_pos - width/2, top_improvements['f1_a'], width, label='Checkpoint A', alpha=0.7)
    ax.bar(x_pos + width/2, top_improvements['f1_b'], width, label='Checkpoint B', alpha=0.7)
    
    ax.set_xlabel('Sample ID (Top 20 Improvements)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Top 20 Samples with Largest F1 Improvement (B vs A)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_improvements['sample_id'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "top_improvements.png", dpi=150, bbox_inches='tight')
    print(f"Saved top improvements to: {output_dir / 'top_improvements.png'}")
    plt.close()
    
    print(f"\n{'='*60}")
    print("Comparison complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    return comparison_df, better_samples_df


def main(
    model_a: str = typer.Option(..., help="Path to model A (checkpoint file or HuggingFace model name)"),
    model_b: str = typer.Option(..., help="Path to model B (checkpoint file or HuggingFace model name)"),
    model_type_a: str = typer.Option(None, help="Model type for A: 'hf_pretrained', 'checkpoint_bigearthnet', or 'checkpoint_multimodal'. Auto-detected if not provided."),
    model_type_b: str = typer.Option(None, help="Model type for B: 'hf_pretrained', 'checkpoint_bigearthnet', or 'checkpoint_multimodal'. Auto-detected if not provided."),
    architecture_a: str = typer.Option(None, help="Architecture for model A (auto-detected if not provided)"),
    architecture_b: str = typer.Option(None, help="Architecture for model B (auto-detected if not provided)"),
    bandconfig_a: str = typer.Option(None, help="Band config for model A (auto-detected if not provided)"),
    bandconfig_b: str = typer.Option(None, help="Band config for model B (auto-detected if not provided)"),
    use_s1_a: bool = typer.Option(None, "--use-s1-a/--no-use-s1-a", help="Whether to include S1 for model A (for multimodal models)"),
    use_s1_b: bool = typer.Option(None, "--use-s1-b/--no-use-s1-b", help="Whether to include S1 for model B (for multimodal models)"),
    seed: int = typer.Option(42, help="Random seed"),
    bs: int = typer.Option(32, help="Batch size"),
    workers: int = typer.Option(8, help="Number of workers"),
    threshold: float = typer.Option(0.5, help="Classification threshold"),
    test_run: bool = typer.Option(False, help="Run with fewer batches for quick testing"),
    output_dir: str = typer.Option("./compare_output", help="Output directory for results"),
    dinov3_model_name_a: str = typer.Option(None, help="DINOv3 HuggingFace model name for model A"),
    dinov3_model_name_b: str = typer.Option(None, help="DINOv3 HuggingFace model name for model B"),
):
    """
    Compare two models and identify where model B performs better than model A.
    
    Supports three model types for each model:
    1. HuggingFace pretrained: Provide model name (e.g., 'hackelle/resnet18-all-v0.1.1')
    2. BigEarthNet checkpoint: Provide path to .ckpt file
    3. Multimodal checkpoint: Provide path to .ckpt file (auto-detected)
    """
    compare_checkpoints(
        model_a_path=model_a,
        model_b_path=model_b,
        model_type_a=model_type_a,
        model_type_b=model_type_b,
        architecture_a=architecture_a,
        architecture_b=architecture_b,
        bandconfig_a=bandconfig_a,
        bandconfig_b=bandconfig_b,
        use_s1_a=use_s1_a,
        use_s1_b=use_s1_b,
        seed=seed,
        bs=bs,
        workers=workers,
        threshold=threshold,
        test_run=test_run,
        output_dir=output_dir,
        dinov3_model_name_a=dinov3_model_name_a,
        dinov3_model_name_b=dinov3_model_name_b,
    )


if __name__ == "__main__":
    typer.run(main)

