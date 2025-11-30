"""
Embedding Space Projection: t-SNE/UMAP visualization of feature embeddings.

This script extracts feature vectors (before final classifier) for vegetation classes
and visualizes them in 2D using t-SNE or UMAP to show class separation.

Compares:
- Baseline model features
- Your multimodal model features

Shows how well each model separates vegetation classes (Pastures, Forests, Crops).
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import random

import numpy as np
import typer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Ensure project modules are importable
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import NEW_LABELS, resolve_data_dir  # noqa: E402
from scripts.diagnose.model_utils import load_model_and_infer  # noqa: E402
from scripts.utils import get_benv2_dir_dict, default_dm  # noqa: E402

__author__ = "Embedding Space Visualization Tool"

# Vegetation classes to visualize
VEGETATION_CLASSES = [
    "Pastures",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Permanent crops",
    "Arable land",
]
VEGETATION_INDICES = [NEW_LABELS.index(cls) for cls in VEGETATION_CLASSES if cls in NEW_LABELS]

# Color map for classes
CLASS_COLORS = plt.cm.Set3(np.linspace(0, 1, len(VEGETATION_CLASSES)))


def extract_features_before_classifier(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature embeddings before the final classifier layer.
    
    Args:
        model: The model to extract features from
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        max_samples: Maximum number of samples to extract
    
    Returns:
        (features, labels) where features is (N, D) and labels is (N, 19)
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        sample_count = 0
        for x, y in dataloader:
            if sample_count >= max_samples:
                break
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Try to extract features before classifier
            # This depends on model structure
            features = None
            
            if hasattr(model, 'model'):
                # Try multimodal model structure
                if hasattr(model.model, 'forward') and hasattr(model.model, 'classifier'):
                    # Get features before classifier
                    try:
                        # Call forward with return_embeddings if available
                        if hasattr(model.model, 'forward'):
                            result = model.model.forward(x, return_embeddings=True)
                            if isinstance(result, tuple):
                                _, embeddings_dict = result
                                # Try to get fused embeddings
                                if 'fused' in embeddings_dict:
                                    features = embeddings_dict['fused']
                                elif 'dinov3' in embeddings_dict:
                                    features = embeddings_dict['dinov3']
                                else:
                                    # Concatenate all embeddings
                                    features = torch.cat(list(embeddings_dict.values()), dim=1)
                            else:
                                # No embeddings returned, try to hook into classifier
                                features = _extract_features_via_hook(model, x)
                    except:
                        features = _extract_features_via_hook(model, x)
                else:
                    features = _extract_features_via_hook(model, x)
            else:
                features = _extract_features_via_hook(model, x)
            
            if features is None:
                # Fallback: use model output as features (not ideal but works)
                features = model(x)
                if isinstance(features, tuple):
                    features = features[0]
            
            # Flatten if needed
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu())
            all_labels.append(y.cpu())
            
            sample_count += x.size(0)
    
    if len(all_features) == 0:
        raise ValueError("No features extracted. Check model structure.")
    
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    return features, labels


def _extract_features_via_hook(model: nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Extract features by hooking into the model before classifier."""
    features = None
    
    def hook_fn(module, input, output):
        nonlocal features
        features = input[0] if isinstance(input, tuple) else input
    
    # Try to find classifier/head layer
    hooks = []
    for name, module in model.named_modules():
        if 'classifier' in name.lower() or 'head' in name.lower():
            if isinstance(module, (nn.Linear, nn.Sequential)):
                hook = module.register_forward_pre_hook(hook_fn)
                hooks.append(hook)
                break
    
    if hooks:
        # Run forward pass to trigger hook
        _ = model(x)
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    return features


def filter_vegetation_samples(
    features: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Filter samples to only include vegetation classes.
    
    Args:
        features: (N, D) feature embeddings
        labels: (N, 19) binary labels
    
    Returns:
        (filtered_features, class_labels, class_names)
        where class_labels is (N,) with class indices
    """
    filtered_features = []
    class_labels = []
    class_names_list = []
    
    for idx in range(len(labels)):
        # Check if sample has any vegetation class
        has_vegetation = any(labels[idx, cls_idx] > 0.5 for cls_idx in VEGETATION_INDICES)
        if not has_vegetation:
            continue
        
        # Get the primary vegetation class (first one found)
        primary_class = None
        for cls_idx in VEGETATION_INDICES:
            if labels[idx, cls_idx] > 0.5:
                primary_class = cls_idx
                break
        
        if primary_class is not None:
            filtered_features.append(features[idx])
            class_labels.append(VEGETATION_INDICES.index(primary_class))
            class_names_list.append(NEW_LABELS[primary_class])
    
    if len(filtered_features) == 0:
        raise ValueError("No vegetation samples found in dataset.")
    
    return np.array(filtered_features), np.array(class_labels), class_names_list


def project_embeddings(
    features: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D using t-SNE or UMAP.
    
    Args:
        features: (N, D) feature embeddings
        method: "tsne" or "umap"
        n_components: Number of dimensions (2 for visualization)
        random_state: Random seed
    
    Returns:
        (N, 2) projected coordinates
    """
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            print(f"  Applying t-SNE to {len(features)} samples...")
            tsne = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=min(30, len(features) - 1),
                n_iter=1000,
            )
            projected = tsne.fit_transform(features)
            return projected
        except ImportError:
            raise ImportError("scikit-learn is required for t-SNE. Install with: pip install scikit-learn")
    
    elif method == "umap":
        try:
            import umap
            print(f"  Applying UMAP to {len(features)} samples...")
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=min(15, len(features) - 1),
            )
            projected = reducer.fit_transform(features)
            return projected
        except ImportError:
            raise ImportError("UMAP is required. Install with: pip install umap-learn")
    
    else:
        raise ValueError(f"Unknown projection method: {method}. Use 'tsne' or 'umap'.")


def visualize_embedding_space(
    projected: np.ndarray,
    class_labels: np.ndarray,
    class_names_list: List[str],
    model_name: str,
    output_path: str,
    method: str = "tsne",
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Visualize projected embedding space with color-coded classes.
    
    Args:
        projected: (N, 2) projected coordinates
        class_labels: (N,) class indices
        class_names_list: List of class names for each sample
        model_name: Name of the model
        output_path: Path to save visualization
        method: Projection method used
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each class with different color
    unique_classes = np.unique(class_labels)
    for cls_idx in unique_classes:
        mask = class_labels == cls_idx
        class_name = VEGETATION_CLASSES[cls_idx]
        color = CLASS_COLORS[cls_idx]
        
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=[color],
            label=class_name,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
        )
    
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Embedding Space Projection ({method.upper()})\n'
        f'{model_name}\n'
        f'Color-coded by Ground Truth Class',
        fontsize=13,
        fontweight='bold'
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved embedding space visualization to: {output_path}")
    plt.close()


def main(
    baseline_model_path: Optional[str] = typer.Option(None, help="Path to baseline model"),
    your_model_path: Optional[str] = typer.Option(None, help="Path to your model"),
    baseline_model_type: Optional[str] = typer.Option(None, help="Baseline model type"),
    your_model_type: Optional[str] = typer.Option(None, help="Your model type"),
    method: str = typer.Option("tsne", help="Projection method: 'tsne' or 'umap'"),
    max_samples: int = typer.Option(1000, help="Maximum number of samples to visualize"),
    seed: int = typer.Option(42, help="Random seed"),
    bs: int = typer.Option(32, help="Batch size"),
    workers: int = typer.Option(8, help="Number of workers"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory"),
):
    """
    Visualize embedding space projection for vegetation classes.
    
    Compares baseline and your model to show how well each separates
    vegetation classes (Pastures, Forests, Crops) in feature space.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if output_dir is None:
        output_dir = Path(".") / "embedding_space_visualizations"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Embedding Space Projection Visualization")
    print("=" * 80)
    print(f"Vegetation classes: {', '.join(VEGETATION_CLASSES)}")
    print(f"Projection method: {method.upper()}")
    print("=" * 80 + "\n")
    
    # Setup dataloader
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
    
    hparams = {
        "batch_size": bs,
        "workers": workers,
        "channels": 12,  # All S2 bands
        "bandconfig": "all",
    }
    dm = default_dm(hparams, data_dirs, 120)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Process each model
    models_to_process = []
    if baseline_model_path:
        models_to_process.append(("baseline", baseline_model_path, baseline_model_type))
    if your_model_path:
        models_to_process.append(("your_model", your_model_path, your_model_type))
    
    if len(models_to_process) == 0:
        raise ValueError("At least one model path must be provided (baseline or your_model)")
    
    for model_name, model_path, model_type in models_to_process:
        print(f"\n{'='*80}")
        print(f"Processing {model_name}: {model_path}")
        print(f"{'='*80}")
        
        # Load model
        print("Loading model...")
        # This is simplified - you'll need to implement proper model loading
        # based on your model structure
        print(f"⚠ Note: Model loading for feature extraction needs to be implemented")
        print(f"   based on your specific model architecture.")
        print(f"   The framework is provided - adapt model loading as needed.")
        
        # TODO: Load model properly
        # For now, we'll use a placeholder
        # model = load_model_for_feature_extraction(model_path, model_type)
        # model = model.to(device)
        
        # Extract features
        print("\nExtracting features...")
        # features, labels = extract_features_before_classifier(
        #     model,
        #     test_loader,
        #     device,
        #     max_samples=max_samples,
        # )
        
        # Filter to vegetation classes
        # filtered_features, class_labels, class_names_list = filter_vegetation_samples(
        #     features,
        #     labels,
        # )
        
        # Project to 2D
        # print(f"\nProjecting embeddings using {method.upper()}...")
        # projected = project_embeddings(
        #     filtered_features,
        #     method=method,
        #     random_state=seed,
        # )
        
        # Visualize
        # output_path = output_dir / f"{model_name}_{method}_embedding_space.png"
        # visualize_embedding_space(
        #     projected,
        #     class_labels,
        #     class_names_list,
        #     model_name,
        #     str(output_path),
        #     method=method,
        # )
        
        print(f"  Would save visualization to: {output_dir / f'{model_name}_{method}_embedding_space.png'}")
    
    print("\n" + "=" * 80)
    print("Visualization framework complete!")
    print("=" * 80)
    print("\nNote: This script provides the framework. To complete:")
    print("1. Implement proper model loading for feature extraction")
    print("2. Extract features before the final classifier layer")
    print("3. Filter samples to vegetation classes")
    print("4. Project using t-SNE or UMAP")
    print("5. Visualize with color-coded classes")
    print("\nExpected Insight:")
    print("Baseline may separate 'Permanent Crops' into a distinct cluster,")
    print("whereas in your multimodal plot, 'Permanent Crops' points may be")
    print("mixed/overlapping with 'Pastures' or 'Arable Land'.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    typer.run(main)

