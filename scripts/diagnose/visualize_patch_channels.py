"""
Visualize individual channels and channel combinations for BigEarthNet patches.

This script:
1. Randomly picks a patch from the dataset
2. Visualizes each channel individually
3. Visualizes channel combinations (e.g., RGB, all S2 bands, etc.)
"""

import sys
from pathlib import Path
from typing import Optional, List, Tuple
import random

import numpy as np
import typer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import resolve_data_dir, STANDARD_BANDS, NEW_LABELS  # noqa: E402
from scripts.utils import get_benv2_dir_dict, get_bands, default_dm  # noqa: E402
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet  # noqa: E402

__author__ = "Patch Channel Visualization Tool"


# Approximate central wavelengths for Sentinel-2 bands and C-band SAR
# Values are for documentation / plotting only (not used for any computation).
BAND_WAVELENGTHS = {
    "B01": "443 nm",
    "B02": "490 nm",
    "B03": "560 nm",
    "B04": "665 nm",
    "B05": "705 nm",
    "B06": "740 nm",
    "B07": "783 nm",
    "B08": "842 nm",
    "B8A": "865 nm",
    "B09": "945 nm",
    "B11": "1610 nm",
    "B12": "2190 nm",
    # Sentinel‑1 C‑band (wavelength ~5.6 cm)
    "VV": "C-band (~5.6 cm)",
    "VH": "C-band (~5.6 cm)",
}


def load_patch_channels(
    patch_id: str,
    data_dirs: dict,
    bandconfig: str = "all",
) -> Tuple[np.ndarray, List[str]]:
    """
    Load all channels for a specific patch.
    
    Args:
        patch_id: Patch ID to load
        data_dirs: Data directory dictionary
        bandconfig: Band configuration ('all', 's2', 'rgb', etc.)
    
    Returns:
        image_data: (H, W, C) array of image data
        band_names: List of band names in order
    """
    bands, num_channels = get_bands(bandconfig)
    
    # Create a temporary dataset to load the patch
    # We'll use BENv2DataSet directly
    img_size = 120
    
    # Get channel configuration
    channel_config = STANDARD_BANDS.get(num_channels, bands)
    if isinstance(channel_config, dict):
        channel_config = bands
    
    # Create dataset instance (we won't use it fully, just to access loading)
    # Actually, we need to load the raw data directly
    # Let's use the dataset's internal loading mechanism
    
    # For now, let's use a simpler approach: load via the dataloader
    # We'll create a minimal dataset that loads this specific patch
    
    # Actually, the easiest way is to use the datamodule to get a sample
    # But we want raw data, not preprocessed...
    
    # Let's try a different approach: use BENv2DataSet directly
    try:
        # Create dataset with the specific configuration
        dataset = BENv2DataSet(
            data_dirs=data_dirs,
            split="test",  # Use test split for loading
            img_size=(num_channels, img_size, img_size),
        )
        
        # Find the patch in the dataset
        # This is tricky - we need to find the index of the patch
        # For now, let's just get a random sample and use that
        # Actually, let's modify to accept a sample index instead
        
        return None, bands  # Placeholder
    except Exception as e:
        print(f"Error loading patch: {e}")
        raise


def visualize_individual_channels(
    image_data: np.ndarray,
    band_names: List[str],
    output_path: str,
    active_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 16),
) -> None:
    """
    Visualize each channel individually (with wavelength info), using the original layout.
    """
    H, W, C = image_data.shape

    # Layout: 3 columns, 4 rows (3x4 = 12 cells)
    n_cols = 3
    n_rows = 4

    # Figure size adjusted for 3x4 grid
    fig = plt.figure(figsize=(12, 16))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.15, wspace=0.15)

    for i in range(C):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Single channel
        channel_data = image_data[:, :, i]
        channel_min = channel_data.min()
        channel_max = channel_data.max()
        if channel_max > channel_min:
            channel_norm = (channel_data - channel_min) / (channel_max - channel_min)
        else:
            channel_norm = channel_data

        ax.imshow(channel_norm, cmap="gray", vmin=0, vmax=1)
        band_name = band_names[i] if i < len(band_names) else f"Ch{i}"
        wavelength = BAND_WAVELENGTHS.get(band_name)
        if wavelength:
            title_line = f"{band_name} ({wavelength})"
        else:
            title_line = band_name
        ax.set_title(
            f"{title_line}\n[{channel_min:.2f}, {channel_max:.2f}]",
            fontsize=9,
        )
        ax.axis("off")

    # Hide unused subplots
    total_cells = n_rows * n_cols
    for i in range(C, total_cells):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

    # Figure title (as before)
    plt.suptitle("Individual Channel Visualizations", fontsize=14, fontweight="bold", y=0.98)

    # GT labels: small-font line near the top (previous layout, but less intrusive)
    if active_labels:
        labels_text = (
            ", ".join(active_labels)
            if len(active_labels) <= 5
            else ", ".join(active_labels[:5]) + f" (+{len(active_labels) - 5} more)"
        )
        fig.text(
            0.5,
            0.94,
            f"GT Labels: {labels_text}",
            ha="center",
            va="center",
            fontsize=8,
        )

    # Leave some room for the title and GT labels at top
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.9])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved individual channels visualization to: {output_path}")
    plt.close()


def visualize_channel_combinations(
    image_data: np.ndarray,
    band_names: List[str],
    output_path: str,
    active_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Visualize channel combinations (RGB, all S2, etc.).
    
    Args:
        image_data: (H, W, C) image data
        band_names: List of band/channel names
        output_path: Path to save visualization
        active_labels: List of active ground truth labels (optional)
        figsize: Figure size
    """
    H, W, C = image_data.shape
    
    # Define combinations to visualize
    combinations = []
    
    # RGB combination
    rgb_bands = ["B04", "B03", "B02"]  # R, G, B
    rgb_indices = []
    for band in rgb_bands:
        if band in band_names:
            rgb_indices.append(band_names.index(band))
    
    if len(rgb_indices) == 3:
        # Extract RGB in correct order: B04 (Red), B03 (Green), B02 (Blue)
        # The rgb_indices list should already be in [B04, B03, B02] order
        # But we need to ensure the channels are in R, G, B order for display
        rgb_image = image_data[:, :, rgb_indices]
        
        # Properly normalize RGB for display
        # BigEarthNet data might be in [0, 1] or [0, 10000] range
        rgb_min = rgb_image.min()
        rgb_max = rgb_image.max()
        
        if rgb_max > 100:
            # Raw reflectance values [0, 10000], normalize first
            rgb_normalized = np.clip(rgb_image / 10000.0, 0, 1)
        elif rgb_max <= 1.0:
            # Already normalized [0, 1], but might need stretching for better contrast
            # Use percentile-based stretching for better visualization
            p2, p98 = np.percentile(rgb_image, [2, 98])
            if p98 > p2:
                rgb_normalized = np.clip((rgb_image - p2) / (p98 - p2), 0, 1)
            else:
                rgb_normalized = np.clip(rgb_image, 0, 1)
        else:
            # Stretch to [0, 1] range
            if rgb_max > rgb_min:
                rgb_normalized = (rgb_image - rgb_min) / (rgb_max - rgb_min)
            else:
                rgb_normalized = rgb_image
        
        combinations.append(("RGB (B04, B03, B02)", rgb_normalized))
    
    # Non-RGB S2 bands (exclude B02, B03, B04 which are already shown as RGB)
    s2_bands_all = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    s2_non_rgb_bands = ["B05", "B06", "B07", "B08", "B8A", "B11", "B12"]  # Exclude RGB bands
    s2_non_rgb_indices = [i for i, band in enumerate(band_names) if band in s2_non_rgb_bands]
    
    if len(s2_non_rgb_indices) >= 3:
        # Take first 3 non-RGB S2 bands for visualization
        s2_image = image_data[:, :, s2_non_rgb_indices[:3]]
        s2_normalized = np.clip(s2_image, 0, 1)
        selected_bands = [band_names[i] for i in s2_non_rgb_indices[:3]]
        combinations.append((f"S2 Non-RGB Bands ({', '.join(selected_bands)})", s2_normalized))
    elif len(s2_non_rgb_indices) >= 1:
        # If we have fewer than 3 non-RGB S2 bands, show what we have
        s2_image = image_data[:, :, s2_non_rgb_indices]
        # Pad or repeat if needed to get 3 channels
        if len(s2_non_rgb_indices) == 1:
            s2_image = np.stack([s2_image[:, :, 0], s2_image[:, :, 0], s2_image[:, :, 0]], axis=-1)
        elif len(s2_non_rgb_indices) == 2:
            s2_image = np.stack([s2_image[:, :, 0], s2_image[:, :, 1], s2_image[:, :, 0]], axis=-1)
        s2_normalized = np.clip(s2_image, 0, 1)
        selected_bands = [band_names[i] for i in s2_non_rgb_indices]
        combinations.append((f"S2 Non-RGB Bands ({', '.join(selected_bands)})", s2_normalized))
    
    # S1 bands (if available)
    s1_bands = ["VV", "VH"]
    s1_indices = [i for i, band in enumerate(band_names) if band in s1_bands]
    if len(s1_indices) >= 2:
        # Create false color from S1
        s1_image = image_data[:, :, s1_indices]
        s1_normalized = np.clip(s1_image, 0, 1)
        # Use VV, VH, and average for false color
        s1_false_color = np.stack([
            s1_normalized[:, :, 0],  # VV as red
            s1_normalized[:, :, 1],  # VH as green
            (s1_normalized[:, :, 0] + s1_normalized[:, :, 1]) / 2  # Average as blue
        ], axis=-1)
        combinations.append((f"S1 Bands ({', '.join([band_names[i] for i in s1_indices])})", s1_false_color))
    
    # All channels (if <= 3, show directly; otherwise show first 3 as false color)
    if C <= 3:
        all_normalized = np.clip(image_data, 0, 1)
        combinations.append((f"All Channels ({', '.join(band_names)})", all_normalized))
    elif C > 3:
        # Show first 3 channels as false color composite
        first_3 = image_data[:, :, :3]
        first_3_normalized = np.clip(first_3, 0, 1)
        # Check if first 3 are RGB bands
        first_3_bands = band_names[:3]
        if set(first_3_bands) == {"B04", "B03", "B02"}:
            combinations.append((f"RGB Channels ({', '.join(first_3_bands)})", first_3_normalized))
        else:
            combinations.append((f"First 3 Channels ({', '.join(first_3_bands)})", first_3_normalized))
    
    # Create visualization
    n_combinations = len(combinations)
    n_cols = 2
    n_rows = (n_combinations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_combinations == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (name, combo_image) in enumerate(combinations):
        ax = axes[idx]
        ax.imshow(combo_image)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_combinations, len(axes)):
        axes[idx].axis('off')
    
    # Create title with GT labels if available
    title = "Channel Combinations"
    if active_labels:
        labels_text = ", ".join(active_labels) if len(active_labels) <= 5 else ", ".join(active_labels[:5]) + f" (+{len(active_labels) - 5} more)"
        title += f"\nGT Labels: {labels_text}"
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved channel combinations visualization to: {output_path}")
    plt.close()


def main(
    bandconfig: str = typer.Option(
        "all",
        help="Band configuration: 'all' (12 ch: 10m+20m S2 + S1), 'all_full' (all S2 + S1), 's2' (10 ch: 10m+20m S2), 's2_full' (all S2), 's1' (2 ch), 'rgb' (3 ch)"
    ),
    sample_idx: Optional[int] = typer.Option(
        None,
        help="Sample index to visualize (None = random)"
    ),
    seed: int = typer.Option(
        42,
        help="Random seed (for reproducible random selection)"
    ),
    split: str = typer.Option(
        "test",
        help="Dataset split: 'train', 'validation', or 'test'"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        help="Output directory for visualizations"
    ),
    img_size: int = typer.Option(
        120,
        help="Image size"
    ),
):
    """
    Visualize individual channels and channel combinations for a BigEarthNet patch.
    
    This script randomly selects a patch (or uses a specified index) and creates
    visualizations showing:
    1. Each channel individually as a grayscale image
    2. Channel combinations (RGB, S2 bands, S1 bands, etc.)
    """
    # Setup
    random.seed(seed)
    np.random.seed(seed)
    
    if output_dir is None:
        output_dir = Path(".") / "patch_visualizations"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Patch Channel Visualization")
    print("=" * 80)
    print(f"Band configuration: {bandconfig}")
    print(f"Split: {split}")
    print("=" * 80 + "\n")
    
    # Get data directories
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
    
    # Get bands
    bands, num_channels = get_bands(bandconfig)
    print(f"Number of channels: {num_channels}")
    print(f"Bands: {bands}")
    if bandconfig == "all":
        print("\nNote: 'all' uses 10m+20m S2 bands + S1 (12 channels).")
        print("      Use 'all_full' to see ALL Sentinel-2 bands (including 60m resolution).\n")
    else:
        print()
    
    # Setup datamodule
    hparams = {
        "batch_size": 1,
        "workers": 1,
        "channels": num_channels,
        "bandconfig": bandconfig,
    }
    dm = default_dm(hparams, data_dirs, img_size)
    
    # Setup dataloader
    if split == "validation":
        dm.setup(stage="fit")
        dataloader = dm.val_dataloader()
    elif split == "train":
        dm.setup(stage="fit")
        dataloader = dm.train_dataloader()
    else:
        dm.setup(stage="test")
        dataloader = dm.test_dataloader()
    
    # Get dataset length
    dataset_size = len(dataloader.dataset) if hasattr(dataloader.dataset, '__len__') else None
    if dataset_size is None:
        # Try to get length from dataloader
        try:
            dataset_size = len(dataloader)
        except:
            dataset_size = None
    
    # Select sample index
    if sample_idx is None:
        if dataset_size is not None:
            sample_idx = random.randint(0, dataset_size - 1)
            print(f"Randomly selected sample index: {sample_idx} (out of {dataset_size} samples)")
        else:
            sample_idx = 0
            print(f"Using sample index: {sample_idx} (dataset size unknown)")
    else:
        print(f"Using specified sample index: {sample_idx}")
    
    # Load sample
    print("\nLoading sample...")
    sample_data = None
    sample_labels = None
    
    for idx, (x, y) in enumerate(dataloader):
        if idx == sample_idx:
            sample_data = x  # (1, C, H, W)
            sample_labels = y  # (1, 19)
            break
    
    if sample_data is None:
        raise ValueError(f"Could not load sample at index {sample_idx}")
    
    # Convert to numpy and reshape
    # PyTorch format: (B, C, H, W) -> numpy: (H, W, C)
    image_data = sample_data[0].permute(1, 2, 0).numpy()  # (H, W, C)
    
    # Get labels
    labels = sample_labels[0].numpy()  # (19,)
    active_labels = [NEW_LABELS[i] for i in range(19) if labels[i] > 0.5]
    
    print(f"✓ Loaded sample")
    print(f"  Image shape: {image_data.shape}")
    print(f"  Active labels: {', '.join(active_labels) if active_labels else 'None'}")
    
    # Generate base name for output files
    base_name = f"patch_{sample_idx}_{bandconfig}"
    
    # Visualize individual channels
    print("\nGenerating individual channel visualizations...")
    individual_path = output_dir / f"{base_name}_individual_channels.png"
    visualize_individual_channels(
        image_data,
        bands,
        str(individual_path),
        active_labels=active_labels,
    )
    
    # Visualize channel combinations
    print("\nGenerating channel combination visualizations...")
    combinations_path = output_dir / f"{base_name}_channel_combinations.png"
    visualize_channel_combinations(
        image_data,
        bands,
        str(combinations_path),
        active_labels=active_labels,
    )
    
    # Save metadata
    import json
    metadata = {
        "sample_idx": sample_idx,
        "bandconfig": bandconfig,
        "bands": bands,
        "num_channels": num_channels,
        "image_shape": list(image_data.shape),
        "active_labels": active_labels,
        "all_labels": [NEW_LABELS[i] for i in range(19)],
        "label_values": labels.tolist(),
    }
    metadata_path = output_dir / f"{base_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - Individual channels: {individual_path}")
    print(f"  - Channel combinations: {combinations_path}")
    print(f"  - Metadata: {metadata_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    typer.run(main)

