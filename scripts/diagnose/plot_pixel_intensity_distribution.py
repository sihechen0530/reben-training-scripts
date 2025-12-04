"""
Plot pixel intensity distribution comparison between BigEarthNet and natural images (DINOv3 training data).

This script:
1. Loads a sample batch from BigEarthNet (RGB images)
2. Simulates "Natural Image" distribution (DINOv3 proxy using ImageNet statistics)
3. Calculates Wasserstein Distance (Earth Mover's Distance) to quantify distribution shift
4. Visualizes the pixel intensity histograms for comparison

Natural images (DINOv3) usually have a bell-curve-like distribution across the 0-255 range.
Satellite imagery often has a "long tail" (high reflectance clouds) and a compressed dynamic range for ground features.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path
import random
from scipy.stats import wasserstein_distance

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import resolve_data_dir  # noqa: E402
from scripts.utils import get_benv2_dir_dict, default_dm  # noqa: E402

__author__ = "Pixel Intensity Distribution Analysis Tool"


def load_bigearthnet_pixels(
    data_dirs: dict,
    split: str = "train",
    num_samples: int = 100,
    img_size: int = 120,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Load pixel values from BigEarthNet RGB images.
    
    Args:
        data_dirs: Data directory dictionary
        split: Dataset split ('train', 'validation', or 'test')
        num_samples: Number of samples to load
        img_size: Image size
        random_seed: Random seed for reproducibility
    
    Returns:
        Flattened array of pixel values (0-1 float range, clipped to handle outliers)
    """
    np.random.seed(random_seed)
    
    # Setup datamodule for RGB (3 channels)
    hparams = {
        "batch_size": 32,
        "workers": 4,
        "channels": 3,
        "bandconfig": "rgb",
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
    
    # Collect pixel values from batches
    all_pixels = []
    samples_collected = 0
    
    print(f"Loading {num_samples} samples from {split} split...")
    for batch_idx, (x, y) in enumerate(dataloader):
        # x shape: (B, C, H, W)
        # BigEarthNet data is typically normalized to [0, 1] in the dataloader
        if isinstance(x, torch.Tensor):
            batch_np = x.cpu().numpy()  # (B, C, H, W)
        else:
            batch_np = np.array(x)  # (B, C, H, W)
        
        # Keep data in 0-1 float range (don't convert to 0-255)
        # Clip outliers: allow slightly >1.0 for specular highlights, but clip extreme outliers
        batch_np = np.clip(batch_np, 0.0, 1.5).astype(np.float32)
        
        # Flatten to get all pixel values: (B, C, H, W) -> (B*C*H*W,)
        batch_pixels = batch_np.flatten()
        all_pixels.append(batch_pixels)
        
        samples_collected += batch_np.shape[0]
        if samples_collected >= num_samples:
            break
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Loaded {samples_collected} samples...")
    
    # Concatenate all pixels
    ben_pixels = np.concatenate(all_pixels, axis=0)
    print(f"✓ Collected {len(ben_pixels)} pixel values from {samples_collected} samples")
    
    return ben_pixels


def load_natural_images_from_directory(
    image_dir: str,
    num_samples: int = 100,
    img_size: int = 120,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Load pixel values from actual natural images (e.g., ImageNet, DINOv3 training data).
    
    Supports:
    - ImageNet structure: image_dir/class_name/*.jpg
    - Flat directory: image_dir/*.jpg (or .png, etc.)
    
    Args:
        image_dir: Directory containing natural images
        num_samples: Number of images to load
        img_size: Target image size (will resize if needed)
        random_seed: Random seed for reproducibility
    
    Returns:
        Flattened array of pixel values (0-255 range)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = []
    
    # Check if it's ImageNet structure (subdirectories) or flat
    subdirs = [d for d in image_dir.iterdir() if d.is_dir()]
    if subdirs:
        # ImageNet structure: image_dir/class_name/*.jpg
        print(f"Detected ImageNet-style structure with {len(subdirs)} classes")
        for subdir in subdirs:
            for ext in image_extensions:
                image_files.extend(list(subdir.glob(f"*{ext}")))
                image_files.extend(list(subdir.glob(f"*{ext.upper()}")))
    else:
        # Flat directory
        print("Detected flat directory structure")
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
            image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    print(f"Found {len(image_files)} image files")
    
    # Randomly sample if we have more than needed
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    all_pixels = []
    loaded_count = 0
    
    print(f"Loading {len(image_files)} images...")
    for img_path in image_files:
        try:
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1] range
            img_array = np.array(img, dtype=np.float32)  # (H, W, 3), values in [0, 255]
            img_array = img_array / 255.0  # Normalize to [0, 1]
            
            # Flatten to get all pixel values
            pixels = img_array.flatten()
            all_pixels.append(pixels)
            loaded_count += 1
            
            if loaded_count % 10 == 0:
                print(f"  Loaded {loaded_count}/{len(image_files)} images...")
        except Exception as e:
            print(f"  Warning: Failed to load {img_path}: {e}")
            continue
    
    if not all_pixels:
        raise ValueError("No images could be loaded successfully")
    
    # Concatenate all pixels
    natural_pixels = np.concatenate(all_pixels, axis=0)
    print(f"✓ Collected {len(natural_pixels)} pixel values from {loaded_count} images")
    
    return natural_pixels


def load_imagenet_dataset(
    imagenet_root: str,
    num_samples: int = 100,
    img_size: int = 120,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Load pixel values from ImageNet dataset using torchvision.
    
    Args:
        imagenet_root: Root directory of ImageNet dataset (should contain 'train' and 'val' folders)
        num_samples: Number of images to load
        img_size: Target image size
        random_seed: Random seed for reproducibility
    
    Returns:
        Flattened array of pixel values (0-255 range)
    """
    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError("torchvision is required for ImageNet loading. Install with: pip install torchvision")
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    imagenet_root = Path(imagenet_root)
    train_dir = imagenet_root / "train"
    
    if not train_dir.exists():
        raise ValueError(f"ImageNet train directory not found: {train_dir}")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])
    
    dataset = datasets.ImageFolder(str(train_dir), transform=transform)
    
    # Sample random indices
    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
    else:
        indices = list(range(len(dataset)))
        print(f"Warning: Only {len(dataset)} images available, using all")
    
    all_pixels = []
    print(f"Loading {len(indices)} images from ImageNet...")
    
    for idx, sample_idx in enumerate(indices):
        img_tensor, _ = dataset[sample_idx]  # (C, H, W), values in [0, 1]
        
        # Convert to numpy (already in [0, 1] range from ToTensor)
        img_array = img_tensor.numpy().astype(np.float32)  # (C, H, W)
        
        # Flatten: (C, H, W) -> (C*H*W,)
        pixels = img_array.flatten()
        all_pixels.append(pixels)
        
        if (idx + 1) % 10 == 0:
            print(f"  Loaded {idx + 1}/{len(indices)} images...")
    
    # Concatenate all pixels
    natural_pixels = np.concatenate(all_pixels, axis=0)
    print(f"✓ Collected {len(natural_pixels)} pixel values from {len(indices)} images")
    
    return natural_pixels


def generate_natural_image_proxy(
    num_pixels: int,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic distribution for natural images (DINOv3 training data proxy).
    
    Uses ImageNet-like statistics: Gaussian distribution in [0, 1] range.
    
    Args:
        num_pixels: Number of pixel values to generate
        random_seed: Random seed for reproducibility
    
    Returns:
        Array of pixel values (0-1 float range)
    """
    np.random.seed(random_seed)
    
    # Natural images (ImageNet/DINOv3) typically have:
    # - Bell-curve-like distribution in normalized [0, 1] space
    # - Centered around 0.45 (middle gray in normalized space)
    # - Standard deviation around 0.225
    natural_proxy = np.random.normal(loc=0.45, scale=0.225, size=num_pixels)
    natural_proxy = np.clip(natural_proxy, 0.0, 1.0).astype(np.float32)
    
    return natural_proxy


def plot_pixel_intensity_distribution(
    ben_pixels: np.ndarray,
    natural_pixels: np.ndarray,
    w_dist: float,
    output_path: str,
    bins: int = 50,
    figsize: tuple = (12, 6),
    is_proxy: bool = False,
) -> None:
    """
    Plot comparative histogram of pixel intensity distributions.
    
    Args:
        ben_pixels: BigEarthNet pixel values
        natural_pixels: Natural image pixel values (actual or proxy)
        w_dist: Wasserstein distance between distributions
        output_path: Path to save the plot
        bins: Number of histogram bins
        figsize: Figure size
        is_proxy: Whether natural_pixels is a synthetic proxy (True) or actual data (False)
    """
    plt.figure(figsize=figsize)
    
    # Determine label based on whether it's proxy or real data
    natural_label = 'DINOv3 Training Data (Proxy)' if is_proxy else 'DINOv3 Training Data (ImageNet)'
    
    # Plot histograms with density normalization
    plt.hist(
        natural_pixels,
        bins=bins,
        alpha=0.6,
        label=natural_label,
        density=True,
        color='blue',
        edgecolor='black',
        linewidth=0.5,
    )
    plt.hist(
        ben_pixels,
        bins=bins,
        alpha=0.6,
        label='BigEarthNet v2 (Your Data)',
        density=True,
        color='red',
        edgecolor='black',
        linewidth=0.5,
    )
    
    plt.title(
        f"Pixel Intensity Distribution Mismatch\nWasserstein Distance: {w_dist:.4f}",
        fontsize=14,
        fontweight='bold',
    )
    plt.xlabel("Pixel Value (0-1 normalized)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics text box
    ben_mean = ben_pixels.mean()
    ben_std = ben_pixels.std()
    natural_mean = natural_pixels.mean()
    natural_std = natural_pixels.std()
    
    stats_text = (
        f"BigEarthNet: μ={ben_mean:.1f}, σ={ben_std:.1f}\n"
        f"Natural Images: μ={natural_mean:.1f}, σ={natural_std:.1f}"
    )
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def main(
    split: str = typer.Option(
        "train",
        help="Dataset split: 'train', 'validation', or 'test'"
    ),
    num_samples: int = typer.Option(
        100,
        help="Number of samples to load from BigEarthNet"
    ),
    img_size: int = typer.Option(
        120,
        help="Image size"
    ),
    bins: int = typer.Option(
        50,
        help="Number of histogram bins"
    ),
    output_path: Optional[str] = typer.Option(
        None,
        help="Output path for the plot (default: pixel_intensity_distribution.png)"
    ),
    random_seed: int = typer.Option(
        42,
        help="Random seed for reproducibility"
    ),
    natural_images_dir: Optional[str] = typer.Option(
        None,
        help="Directory containing natural images (ImageNet or other). If provided, will use actual images instead of synthetic proxy. Supports ImageNet structure (subdirs) or flat directory."
    ),
    imagenet_root: Optional[str] = typer.Option(
        None,
        help="Root directory of ImageNet dataset (should contain 'train' folder). If provided, will load from ImageNet using torchvision. Takes precedence over natural_images_dir."
    ),
):
    """
    Plot pixel intensity distribution comparison between BigEarthNet and natural images.
    
    This script generates a comparative histogram showing:
    - BigEarthNet v2 pixel intensity distribution (satellite imagery)
    - DINOv3 training data distribution (natural images - actual ImageNet or synthetic proxy)
    - Wasserstein Distance metric quantifying the distribution shift
    
    You can use actual DINOv3 training data (ImageNet) by providing:
    - --imagenet-root: Path to ImageNet dataset root (uses torchvision)
    - --natural-images-dir: Path to directory with natural images (supports ImageNet structure or flat directory)
    
    If neither is provided, uses a synthetic proxy distribution based on ImageNet statistics.
    """
    print("\n" + "=" * 80)
    print("Pixel Intensity Distribution Analysis")
    print("=" * 80)
    print(f"Split: {split}")
    print(f"Number of samples: {num_samples}")
    print(f"Image size: {img_size}")
    print("=" * 80 + "\n")
    
    # Get data directories
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
    
    # Load BigEarthNet pixel values
    print("Step 1: Loading BigEarthNet images...")
    ben_pixels = load_bigearthnet_pixels(
        data_dirs=data_dirs,
        split=split,
        num_samples=num_samples,
        img_size=img_size,
        random_seed=random_seed,
    )
    
    # Load or generate natural image distribution
    is_proxy = True
    if imagenet_root is not None:
        print(f"\nStep 2: Loading ImageNet dataset from {imagenet_root}...")
        try:
            natural_pixels = load_imagenet_dataset(
                imagenet_root=imagenet_root,
                num_samples=num_samples,
                img_size=img_size,
                random_seed=random_seed,
            )
            is_proxy = False
        except Exception as e:
            print(f"  Warning: Failed to load ImageNet: {e}")
            print(f"  Falling back to synthetic proxy...")
            natural_pixels = generate_natural_image_proxy(
                num_pixels=len(ben_pixels),
                random_seed=random_seed,
            )
    elif natural_images_dir is not None:
        print(f"\nStep 2: Loading natural images from {natural_images_dir}...")
        try:
            natural_pixels = load_natural_images_from_directory(
                image_dir=natural_images_dir,
                num_samples=num_samples,
                img_size=img_size,
                random_seed=random_seed,
            )
            is_proxy = False
        except Exception as e:
            print(f"  Warning: Failed to load natural images: {e}")
            print(f"  Falling back to synthetic proxy...")
            natural_pixels = generate_natural_image_proxy(
                num_pixels=len(ben_pixels),
                random_seed=random_seed,
            )
    else:
        print(f"\nStep 2: Generating natural image proxy distribution...")
        natural_pixels = generate_natural_image_proxy(
            num_pixels=len(ben_pixels),
            random_seed=random_seed,
        )
    
    print(f"✓ Collected {len(natural_pixels)} pixel values")
    
    # Step 3: 2%-98% robust scaling for BigEarthNet pixels with masking
    print(f"\nStep 3: Applying 2%-98% robust scaling with masking to BigEarthNet pixels...")
    # Mask out padding/nodata (absolute blacks)
    valid_pixels = ben_pixels[ben_pixels > 1e-4]
    if valid_pixels.size > 100:
        p2, p98 = np.percentile(valid_pixels, [2, 98])
    else:
        p2, p98 = 0.0, 1.0

    if p98 <= p2 + 1e-8:
        # Fallback to simple clipping if percentiles are degenerate
        ben_pixels_scaled = np.clip(ben_pixels, 0.0, 1.0).astype(np.float32)
        print("  Warning: Degenerate percentiles, falling back to simple [0,1] clipping.")
    else:
        ben_pixels_scaled = (ben_pixels - p2) / (p98 - p2 + 1e-6)
        ben_pixels_scaled = np.clip(ben_pixels_scaled, 0.0, 1.0).astype(np.float32)
    
    # Apply gamma correction (same as HybridSatelliteNormalizer) after robust scaling
    gamma = 1.0 / 2.2
    ben_pixels_gamma = np.power(ben_pixels_scaled, gamma).astype(np.float32)
    
    # Calculate Wasserstein Distance on gamma-corrected pixels
    print(f"\nStep 4: Calculating Wasserstein Distance (robust + gamma)...")
    w_dist = wasserstein_distance(ben_pixels_gamma, natural_pixels)
    print(f"✓ Real Wasserstein Distance: {w_dist:.4f}")
    print(f"✓ BigEarthMean (robust+gamma): {np.mean(ben_pixels_gamma):.2f} vs NaturalMean: {np.mean(natural_pixels):.2f}")
    
    # Print statistics (using robust-scaled data)
    natural_label = "Natural Images (Proxy)" if is_proxy else "Natural Images (ImageNet)"
    print(f"\nStatistics (after robust scaling + gamma):")
    print(f"  BigEarthNet (robust+gamma to [0,1]):")
    print(f"    Mean: {ben_pixels_gamma.mean():.2f}")
    print(f"    Std:  {ben_pixels_gamma.std():.2f}")
    print(f"    Min:  {ben_pixels_gamma.min():.2f}")
    print(f"    Max:  {ben_pixels_gamma.max():.2f}")
    print(f"  {natural_label}:")
    print(f"    Mean: {natural_pixels.mean():.2f}")
    print(f"    Std:  {natural_pixels.std():.2f}")
    print(f"    Min:  {natural_pixels.min():.2f}")
    print(f"    Max:  {natural_pixels.max():.2f}")
    
    # Generate plot
    print(f"\nStep 5: Generating visualization...")
    if output_path is None:
        output_path = f"pixel_intensity_distribution_{split}_{num_samples}samples.png"
    
    plot_pixel_intensity_distribution(
        ben_pixels=ben_pixels_gamma,
        natural_pixels=natural_pixels,
        w_dist=w_dist,
        output_path=output_path,
        bins=bins,
        is_proxy=is_proxy,
    )
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print(f"Output: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    typer.run(main)

