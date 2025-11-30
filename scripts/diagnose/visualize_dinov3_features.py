"""
Visualize DINOv3 patch embeddings and attention patterns.

Implements three visualization methods:
1. PCA of Patch Embeddings - Projects high-dimensional patch embeddings to RGB
2. CLS Token Attention Map - Shows which patches the CLS token attends to
3. Self-Similarity Maps - Shows which patches are similar to a selected patch

Supports:
- HuggingFace pretrained DINOv3 models
- BigEarthNet checkpoints with DINOv3 backbone
- Multimodal checkpoints with DINOv3 backbone
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List
import warnings

import numpy as np
import typer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom

# Ensure project modules are importable when running from scripts directory
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.diagnose.model_utils import load_model_and_infer  # noqa: E402
from configilm.extra.BENv2_utils import resolve_data_dir  # noqa: E402
from scripts.utils import get_benv2_dir_dict, default_dm  # noqa: E402

__author__ = "DINOv3 Visualization Tool"


def extract_dinov3_backbone(model) -> Optional[nn.Module]:
    """
    Extract DINOv3 backbone from various model types.
    
    Returns:
        DINOv3 backbone model or None if not found
    """
    # Try to find DINOv3 backbone in the model
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        # Check if it's a DINOv3Backbone wrapper
        if hasattr(backbone, 'backbone'):
            return backbone.backbone
        return backbone
    
    # Check for multimodal model structure
    if hasattr(model, 'model'):
        if hasattr(model.model, 'dinov3_backbone'):
            dinov3_backbone = model.model.dinov3_backbone
            if hasattr(dinov3_backbone, 'backbone'):
                return dinov3_backbone.backbone
            return dinov3_backbone
    
    # Direct DINOv3 model
    if hasattr(model, 'vit') or hasattr(model, 'transformer'):
        return model
    
    return None


def get_patch_embeddings_and_attention(
    dinov3_model: nn.Module,
    image: torch.Tensor,
    return_attention: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    """
    Extract patch embeddings and attention weights from DINOv3 model.
    
    Args:
        dinov3_model: DINOv3 model
        image: Input image tensor (B, C, H, W)
        return_attention: Whether to extract attention weights
    
    Returns:
        patch_embeddings: (B, N_patches, D) patch embeddings
        attention_weights: (B, N_heads, N_tokens, N_tokens) or None
        patch_h: Number of patches in height
        patch_w: Number of patches in width
    """
    dinov3_model.eval()
    
    with torch.no_grad():
        # Get image size
        B, C, H, W = image.shape
        
        # Forward pass - try different ways to call the model
        # Set attention implementation to 'eager' if needed (for output_attentions support)
        if return_attention and hasattr(dinov3_model, 'set_attn_implementation'):
            try:
                dinov3_model.set_attn_implementation('eager')
            except Exception:
                pass  # Some models don't support this method
        
        # DINOv3 from transformers supports output_attentions
        try:
            if return_attention:
                outputs = dinov3_model(pixel_values=image, output_attentions=True)
            else:
                outputs = dinov3_model(pixel_values=image)
        except (TypeError, AttributeError):
            try:
                if return_attention:
                    outputs = dinov3_model(image, output_attentions=True)
                else:
                    outputs = dinov3_model(image)
            except (TypeError, AttributeError):
                # Some models don't support output_attentions parameter
                outputs = dinov3_model(image)
                return_attention = False
        
        # Extract last hidden state (patch embeddings)
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state  # (B, N_tokens, D)
        elif isinstance(outputs, torch.Tensor):
            last_hidden = outputs
        elif hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
            last_hidden = outputs.hidden_states[-1]
        else:
            raise ValueError(f"Could not extract hidden states from model output: {type(outputs)}")
        
        # Extract attention if available
        attention_weights = None
        if return_attention:
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # Get attention from last layer
                attention_weights = outputs.attentions[-1]  # (B, N_heads, N_tokens, N_tokens)
            elif hasattr(outputs, 'attention_probs'):
                attention_weights = outputs.attention_probs
        
        # DINOv3 token structure: [CLS] [REG1] [REG2] [REG3] [REG4] [PATCH1] ... [PATCHN]
        # Standard DINOv3 has 4 register tokens, so patches start at index 5
        # However, some variants may have different structures
        
        # Calculate expected number of patches
        # DINOv3 uses patch_size=16 by default
        patch_size = 16
        img_size = max(H, W)
        # Models typically pad to nearest multiple of patch_size
        # For 120x120: pad to 128, so 128/16 = 8 patches per side
        padded_size = ((img_size + patch_size - 1) // patch_size) * patch_size
        n_patches_per_side = padded_size // patch_size
        expected_n_patches = n_patches_per_side * n_patches_per_side
        
        # Find where patches start
        n_tokens = last_hidden.shape[1]
        
        # DINOv3 standard: 1 CLS + 4 registers = 5 special tokens
        # Patches should be: n_tokens - 5
        if n_tokens == expected_n_patches + 5:
            # Standard DINOv3 with 4 register tokens
            patch_start_idx = 5
            n_patch_tokens = expected_n_patches
        elif n_tokens == expected_n_patches + 1:
            # No register tokens, just CLS
            patch_start_idx = 1
            n_patch_tokens = expected_n_patches
        else:
            # Try to infer: assume patches are the majority of tokens
            # and special tokens are at the beginning
            n_special_tokens = n_tokens - expected_n_patches
            if n_special_tokens > 0 and n_special_tokens <= 10:
                patch_start_idx = n_special_tokens
                n_patch_tokens = expected_n_patches
            else:
                # Fallback: use all tokens except first (CLS)
                patch_start_idx = 1
                n_patch_tokens = n_tokens - 1
                # Recalculate grid size
                patch_h = int(np.sqrt(n_patch_tokens))
                patch_w = n_patch_tokens // patch_h
                n_patches_per_side = patch_h
        
        patch_embeddings = last_hidden[:, patch_start_idx:patch_start_idx + n_patch_tokens, :]
        
        # Reshape to spatial grid
        patch_h = patch_w = n_patches_per_side
        if patch_embeddings.shape[1] != n_patch_tokens:
            # Adjust if needed
            actual_n_patches = patch_embeddings.shape[1]
            patch_h = int(np.sqrt(actual_n_patches))
            patch_w = actual_n_patches // patch_h
            if patch_h * patch_w != actual_n_patches:
                # Not a perfect square, use rectangular grid
                patch_w = int(np.ceil(np.sqrt(actual_n_patches)))
                patch_h = (actual_n_patches + patch_w - 1) // patch_w
        
        patch_embeddings = patch_embeddings.reshape(B, patch_h, patch_w, -1)
        
        return patch_embeddings, attention_weights, patch_h, patch_w


def visualize_pca_embeddings(
    patch_embeddings: torch.Tensor,
    patch_h: int,
    patch_w: int,
    output_path: str,
    alpha: float = 0.7,
) -> None:
    """
    Visualize patch embeddings using PCA projection to RGB.
    
    Args:
        patch_embeddings: (B, H, W, D) patch embeddings
        patch_h: Height in patches
        patch_w: Width in patches
        output_path: Path to save visualization
        alpha: Transparency for overlay
    """
    # Take first image in batch
    if len(patch_embeddings.shape) == 4:
        patches = patch_embeddings[0].cpu().numpy()  # (H, W, D)
    else:
        patches = patch_embeddings.cpu().numpy()
    
    # Flatten to (N, D)
    N, D = patches.shape[0] * patches.shape[1], patches.shape[2]
    patches_flat = patches.reshape(N, D)
    
    # Apply PCA to reduce to 3 dimensions (RGB)
    pca = PCA(n_components=3)
    patches_pca = pca.fit_transform(patches_flat)
    
    # Normalize to [0, 1] for RGB
    scaler = MinMaxScaler()
    patches_pca = scaler.fit_transform(patches_pca)
    
    # Reshape back to spatial grid
    patches_rgb = patches_pca.reshape(patch_h, patch_w, 3)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(patches_rgb)
    ax.set_title("PCA of Patch Embeddings\n(Similar colors = Similar semantics)", fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved PCA visualization to: {output_path}")
    plt.close()


def visualize_cls_attention(
    attention_weights: torch.Tensor,
    patch_h: int,
    patch_w: int,
    output_path: str,
    head_idx: Optional[int] = None,
    overlay_image: Optional[np.ndarray] = None,
) -> None:
    """
    Visualize CLS token attention map, optionally overlaid on original image.
    
    Args:
        attention_weights: (B, N_heads, N_tokens, N_tokens) attention weights
        patch_h: Height in patches
        patch_w: Width in patches
        output_path: Path to save visualization
        head_idx: Which attention head to visualize (None = average all heads)
        overlay_image: Optional RGB image to overlay attention on (H, W, 3)
    """
    # Take first image in batch
    attn = attention_weights[0].cpu().numpy()  # (N_heads, N_tokens, N_tokens)
    
    # CLS token is at index 0, get its attention to all other tokens
    cls_attn = attn[:, 0, :]  # (N_heads, N_tokens)
    
    # Average across heads or use specific head
    if head_idx is not None:
        cls_attn = cls_attn[head_idx:head_idx+1]
    
    cls_attn = cls_attn.mean(axis=0)  # (N_tokens,)
    
    # Skip CLS token itself and register tokens, get only patch tokens
    # Use same logic as in get_patch_embeddings_and_attention
    expected_n_patches = patch_h * patch_w
    n_tokens = len(cls_attn)
    
    if n_tokens == expected_n_patches + 5:
        # Standard DINOv3 with 4 register tokens
        patch_start_idx = 5
    elif n_tokens == expected_n_patches + 1:
        # No register tokens, just CLS
        patch_start_idx = 1
    else:
        # Infer: assume special tokens at beginning
        n_special_tokens = n_tokens - expected_n_patches
        if n_special_tokens > 0 and n_special_tokens <= 10:
            patch_start_idx = n_special_tokens
        else:
            patch_start_idx = 1
            expected_n_patches = n_tokens - patch_start_idx
    
    patch_attn = cls_attn[patch_start_idx:patch_start_idx + expected_n_patches]
    
    # Reshape to spatial grid
    patch_attn = patch_attn.reshape(patch_h, patch_w)
    
    # Upsample attention map to match image size if overlay is provided
    if overlay_image is not None:
        img_h, img_w = overlay_image.shape[:2]
        # Upsample attention map to image size
        zoom_factors = (img_h / patch_h, img_w / patch_w)
        attn_upsampled = zoom(patch_attn, zoom_factors, order=1)
        
        # Normalize attention for overlay
        attn_normalized = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min() + 1e-8)
        
        # Create overlay visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        axes[0].imshow(overlay_image)
        axes[0].set_title("Original RGB Image", fontsize=14)
        axes[0].axis('off')
        
        # Overlay attention on image
        axes[1].imshow(overlay_image)
        im = axes[1].imshow(attn_normalized, cmap='hot', alpha=0.6, interpolation='bilinear')
        axes[1].set_title("CLS Token Attention Overlay\n(Brighter = More attended)", fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved CLS attention overlay to: {output_path}")
        plt.close()
    else:
        # Create visualization without overlay
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.imshow(patch_attn, cmap='hot', interpolation='nearest')
        ax.set_title("CLS Token Attention Map\n(Brighter = More attended)", fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved CLS attention map to: {output_path}")
        plt.close()


def visualize_self_similarity(
    patch_embeddings: torch.Tensor,
    patch_h: int,
    patch_w: int,
    output_path: str,
    selected_patch: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Visualize self-similarity map for a selected patch.
    
    Args:
        patch_embeddings: (B, H, W, D) patch embeddings
        patch_h: Height in patches
        patch_w: Width in patches
        output_path: Path to save visualization
        selected_patch: (row, col) of patch to compare against (None = center patch)
    """
    # Take first image in batch
    if len(patch_embeddings.shape) == 4:
        patches = patch_embeddings[0].cpu().numpy()  # (H, W, D)
    else:
        patches = patch_embeddings.cpu().numpy()
    
    # Flatten to (N, D)
    patches_flat = patches.reshape(-1, patches.shape[-1])
    
    # Select patch to compare against
    if selected_patch is None:
        selected_patch = (patch_h // 2, patch_w // 2)
    
    row, col = selected_patch
    patch_idx = row * patch_w + col
    selected_embedding = patches_flat[patch_idx]  # (D,)
    
    # Compute cosine similarity with all other patches
    # Normalize embeddings
    selected_norm = selected_embedding / (np.linalg.norm(selected_embedding) + 1e-8)
    patches_norm = patches_flat / (np.linalg.norm(patches_flat, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    similarities = np.dot(patches_norm, selected_norm)  # (N,)
    
    # Reshape to spatial grid
    similarity_map = similarities.reshape(patch_h, patch_w)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(similarity_map, cmap='viridis', interpolation='nearest')
    
    # Mark the selected patch
    rect = patches.Rectangle(
        (col - 0.5, row - 0.5), 1, 1,
        linewidth=3, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    
    ax.set_title(
        f"Self-Similarity Map\n(Selected patch at ({row}, {col}), "
        f"Brighter = More similar)",
        fontsize=14
    )
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved self-similarity map to: {output_path}")
    plt.close()


def main(
    model_path: str = typer.Option(
        ...,
        help="Path to checkpoint file or HuggingFace model name. "
        "For DINOv3 visualization, use a model with DINOv3 backbone."
    ),
    model_type: Optional[str] = typer.Option(
        None,
        help="Model type: 'hf_pretrained', 'checkpoint_bigearthnet', or "
        "'checkpoint_multimodal'. Auto-detected if not provided."
    ),
    image_path: Optional[str] = typer.Option(
        None,
        help="Path to a single image file to visualize. If None, will use a sample from the dataset."
    ),
    sample_idx: int = typer.Option(
        0,
        help="Index of sample to visualize if using dataset (default: 0)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        help="Output directory for visualizations. If None, will save to same directory as model."
    ),
    visualize_pca: bool = typer.Option(
        True,
        help="Generate PCA visualization of patch embeddings"
    ),
    visualize_cls_attention: bool = typer.Option(
        True,
        help="Generate CLS token attention map"
    ),
    overlay_attention: bool = typer.Option(
        True,
        help="Overlay attention map on original image (requires image_path or dataset sample)"
    ),
    visualize_similarity: bool = typer.Option(
        True,
        help="Generate self-similarity map"
    ),
    similarity_patch_row: Optional[int] = typer.Option(
        None,
        help="Row index of patch for similarity map (None = center)"
    ),
    similarity_patch_col: Optional[int] = typer.Option(
        None,
        help="Column index of patch for similarity map (None = center)"
    ),
    attention_head: Optional[int] = typer.Option(
        None,
        help="Specific attention head to visualize (None = average all heads)"
    ),
):
    """
    Visualize DINOv3 patch embeddings and attention patterns.
    
    This tool generates three types of visualizations:
    1. PCA of Patch Embeddings - Shows semantic similarity via color
    2. CLS Token Attention Map - Shows which regions the model focuses on
    3. Self-Similarity Map - Shows which patches are similar to a selected patch
    """
    # Setup output directory
    if output_dir is None:
        model_path_obj = Path(model_path)
        if model_path_obj.is_file():
            output_dir = str(model_path_obj.parent)
        else:
            output_dir = "."
        print(f"\nUsing default output directory: {output_dir}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model (we'll extract DINOv3 backbone separately)
    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)
    
    # For visualization, we need to load the model differently to access the backbone
    # We'll use a simplified loading approach
    from scripts.diagnose.model_utils import _detect_model_type
    
    if model_type is None:
        model_type = _detect_model_type(model_path)
    
    print(f"Model Type: {model_type}")
    print(f"Model Path: {model_path}")
    
    # Load the full model first to get the backbone
    import lightning.pytorch as pl
    import torch
    from configilm.ConfigILM import ILMConfiguration, ILMType
    from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
    from multimodal.lightning_module import MultiModalLightningModule
    
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    # Load model based on type
    if model_type == "hf_pretrained":
        try:
            model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            print("Trying workaround...")
            # Use workaround from model_utils
            from huggingface_hub import snapshot_download
            import json
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            local_dir = snapshot_download(repo_id=model_path, cache_dir=cache_dir)
            local_path = Path(local_dir)
            config_file = local_path / "config.json"
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            if isinstance(config_dict, dict):
                if 'config' in config_dict:
                    config_data = config_dict['config']
                else:
                    config_data = config_dict
                config = ILMConfiguration(
                    network_type=ILMType.IMAGE_CLASSIFICATION,
                    classes=config_data.get('classes', 19),
                    image_size=config_data.get('image_size', 120),
                    drop_rate=config_data.get('drop_rate', 0.15),
                    drop_path_rate=config_data.get('drop_path_rate', 0.15),
                    timm_model_name=config_data.get('timm_model_name', 'resnet101'),
                    channels=config_data.get('channels', 10),
                )
                model = BigEarthNetv2_0_ImageClassifier(
                    config=config,
                    lr=config_data.get('lr', 0.001),
                    warmup=config_data.get('warmup', None),
                )
                model_file = local_path / "pytorch_model.bin"
                if not model_file.exists():
                    model_file = local_path / "model.safetensors"
                if model_file.suffix == '.safetensors':
                    from safetensors.torch import load_file
                    state_dict = load_file(str(model_file))
                else:
                    state_dict = torch.load(model_file, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict, strict=False)
        model.eval()
    elif model_type == "checkpoint_bigearthnet":
        ckpt_path = Path(model_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        hparams = checkpoint.get('hyper_parameters', {})
        architecture = hparams.get('architecture', 'resnet101')
        
        # Only proceed if it's a DINOv3 model
        if not architecture.startswith('dinov3'):
            raise ValueError(
                f"Model architecture '{architecture}' is not DINOv3. "
                f"This visualization tool only works with DINOv3 models."
            )
        
        # Load model (simplified - you may need to adjust based on your checkpoint format)
        channels = hparams.get('channels', 3)
        config = ILMConfiguration(
            network_type=ILMType.IMAGE_CLASSIFICATION,
            classes=19,
            image_size=120,
            drop_rate=hparams.get('dropout', 0.15),
            drop_path_rate=hparams.get('drop_path_rate', 0.15),
            timm_model_name=architecture,
            channels=channels,
        )
        model = BigEarthNetv2_0_ImageClassifier.load_from_checkpoint(
            str(ckpt_path),
            config=config,
            lr=0.001,
            warmup=1000,
        )
        model.eval()
    elif model_type == "checkpoint_multimodal":
        ckpt_path = Path(model_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Load multimodal model
        from multimodal.lightning_module import MultiModalLightningModule
        # You'll need to construct the config based on checkpoint
        # This is simplified - adjust as needed
        model = MultiModalLightningModule.load_from_checkpoint(str(ckpt_path))
        model.eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Extract DINOv3 backbone
    print("\nExtracting DINOv3 backbone...")
    dinov3_backbone = extract_dinov3_backbone(model)
    
    if dinov3_backbone is None:
        raise ValueError(
            "Could not extract DINOv3 backbone from model. "
            "Please ensure the model uses a DINOv3 backbone."
        )
    
    print("✓ DINOv3 backbone extracted")
    
    # Load or generate image
    print("\n" + "=" * 80)
    print("Loading Image")
    print("=" * 80)
    
    if image_path is not None:
        # Load from file
        from PIL import Image
        import torchvision.transforms as transforms
        
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(img).unsqueeze(0)  # (1, 3, 120, 120)
        print(f"Loaded image from: {image_path}")
    else:
        # Load sample from dataset
        print(f"Loading sample {sample_idx} from dataset...")
        hostname, data_dirs = get_benv2_dir_dict()
        data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
        
        hparams = {
            "batch_size": 1,
            "workers": 1,
            "channels": 3,  # RGB for visualization
            "bandconfig": "rgb",
        }
        dm = default_dm(hparams, data_dirs, 120)
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()
        
        # Get specific sample
        for idx, (x, y) in enumerate(test_loader):
            if idx == sample_idx:
                image_tensor = x  # (1, C, H, W)
                print(f"✓ Loaded sample {sample_idx}")
                break
        else:
            raise ValueError(f"Sample index {sample_idx} not found in dataset")
    
    device = next(dinov3_backbone.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Extract features
    print("\n" + "=" * 80)
    print("Extracting Patch Embeddings and Attention")
    print("=" * 80)
    
    patch_embeddings, attention_weights, patch_h, patch_w = get_patch_embeddings_and_attention(
        dinov3_backbone,
        image_tensor,
        return_attention=visualize_cls_attention,
    )
    
    print(f"✓ Extracted patch embeddings: {patch_embeddings.shape}")
    if attention_weights is not None:
        print(f"✓ Extracted attention weights: {attention_weights.shape}")
    print(f"✓ Patch grid: {patch_h} x {patch_w}")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    base_name = Path(model_path).stem if Path(model_path).is_file() else model_path.replace("/", "_")
    
    if visualize_pca:
        pca_path = output_dir / f"{base_name}_pca_embeddings.png"
        visualize_pca_embeddings(patch_embeddings, patch_h, patch_w, str(pca_path))
    
    if visualize_cls_attention and attention_weights is not None:
        cls_path = output_dir / f"{base_name}_cls_attention.png"
        
        # Prepare overlay image if requested
        overlay_img = None
        if overlay_attention:
            # Get original RGB image for overlay
            if image_path is not None:
                from PIL import Image
                img = Image.open(image_path).convert('RGB')
                overlay_img = np.array(img)
            else:
                # Extract RGB from the loaded tensor (first 3 channels)
                if image_tensor.shape[1] >= 3:
                    # Take RGB channels and convert to numpy
                    rgb_tensor = image_tensor[0, :3, :, :].cpu()
                    # Denormalize if needed (assuming ImageNet normalization)
                    # For BigEarthNet, images might not be normalized, so just convert
                    overlay_img = rgb_tensor.permute(1, 2, 0).numpy()
                    # Clip to [0, 1] and convert to [0, 255]
                    overlay_img = np.clip(overlay_img, 0, 1)
                    if overlay_img.max() <= 1.0:
                        overlay_img = (overlay_img * 255).astype(np.uint8)
                    else:
                        overlay_img = overlay_img.astype(np.uint8)
        
        visualize_cls_attention(
            attention_weights,
            patch_h,
            patch_w,
            str(cls_path),
            head_idx=attention_head,
            overlay_image=overlay_img,
        )
    elif visualize_cls_attention:
        print("⚠ Attention weights not available, skipping CLS attention visualization")
    
    if visualize_similarity:
        similarity_path = output_dir / f"{base_name}_self_similarity.png"
        selected_patch = None
        if similarity_patch_row is not None and similarity_patch_col is not None:
            selected_patch = (similarity_patch_row, similarity_patch_col)
        visualize_self_similarity(
            patch_embeddings,
            patch_h,
            patch_w,
            str(similarity_path),
            selected_patch=selected_patch,
        )
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    typer.run(main)

