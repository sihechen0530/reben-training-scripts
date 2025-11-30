"""
Success/Failure Visualization: Compare models and analyze misclassifications.

This script can visualize two scenarios:

1. "Why We Won" - Success cases where:
   - Ground Truth = Target classes (e.g., Industrial, Coastal Wetlands)
   - Baseline Prediction = False (or low confidence)
   - Your Model Prediction = True (High confidence)
   - Visualizes: DINOv3 attention vs ResNet Grad-CAM

2. "Why We Lost" - Failure cases where:
   - Ground Truth = Target class (e.g., Permanent Crops)
   - Your Model Prediction = Wrong class
   - Visualizes: Spectral profiles showing RGB vs NIR/Red-Edge differences
"""

import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import random

import numpy as np
import typer
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom

# Ensure project modules are importable
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.extra.BENv2_utils import NEW_LABELS, resolve_data_dir, STANDARD_BANDS  # noqa: E402
from configilm.ConfigILM import ILMConfiguration, ILMType  # noqa: E402
from scripts.diagnose.model_utils import (  # noqa: E402
    load_model_and_infer,
    _detect_model_type,
    _infer_dinov3_model_from_checkpoint,
)
from scripts.diagnose.visualize_dinov3_features import (  # noqa: E402
    extract_dinov3_backbone,
    get_patch_embeddings_and_attention,
    visualize_cls_attention,
)
from scripts.utils import get_benv2_dir_dict, get_bands, default_dm  # noqa: E402
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier  # noqa: E402
from multimodal.lightning_module import MultiModalLightningModule  # noqa: E402

__author__ = "Success/Failure Visualization Tool"

# Sentinel-2 band wavelengths (nm) - approximate center wavelengths
S2_WAVELENGTHS = {
    "B02": 490,   # Blue
    "B03": 560,   # Green
    "B04": 665,   # Red
    "B05": 705,   # Red Edge 1
    "B06": 740,   # Red Edge 2
    "B07": 783,   # Red Edge 3
    "B08": 842,   # NIR
    "B8A": 865,   # Red Edge 4
    "B11": 1610,  # SWIR 1
    "B12": 2190,  # SWIR 2
}

# Sentinel-1 bands (no wavelength, use placeholder)
S1_WAVELENGTHS = {
    "VV": 0,  # C-band SAR
    "VH": 0,
}


def find_class_index(class_name_variants: List[str]) -> Tuple[int, str]:
    """Find class index by trying multiple name variations."""
    for variant in class_name_variants:
        try:
            return NEW_LABELS.index(variant), variant
        except ValueError:
            continue
    # Try case-insensitive search
    for variant in class_name_variants:
        for i, label in enumerate(NEW_LABELS):
            if variant.lower() in label.lower() or label.lower() in variant.lower():
                return i, label
    raise ValueError(f"Could not find class. Tried: {class_name_variants}. Available: {NEW_LABELS}")


def parse_class_names(class_names_str: str) -> List[Tuple[int, str]]:
    """
    Parse comma-separated class names and return their indices.
    
    Args:
        class_names_str: Comma-separated class names
    
    Returns:
        List of (class_index, class_name) tuples
    """
    class_names = [name.strip() for name in class_names_str.split(",")]
    result = []
    for name in class_names:
        # Try common variations
        variants = [name]
        if "industrial" in name.lower():
            variants.extend(["Industrial or commercial units", "Industrial", "Industrial units"])
        elif "coastal" in name.lower() or "wetland" in name.lower():
            variants.extend(["Coastal wetlands", "Inland wetlands", "Marine waters", "Inland waters"])
        elif "permanent" in name.lower() and "crop" in name.lower():
            variants.extend(["Permanent crops"])
        
        idx, actual_name = find_class_index(variants)
        result.append((idx, actual_name))
    return result


def get_band_wavelengths(band_names: List[str]) -> List[float]:
    """Get wavelengths for a list of band names."""
    wavelengths = []
    for band in band_names:
        if band in S2_WAVELENGTHS:
            wavelengths.append(S2_WAVELENGTHS[band])
        elif band in S1_WAVELENGTHS:
            wavelengths.append(3000 + len([w for w in wavelengths if w > 2000]))
        else:
            wavelengths.append(0)
    return wavelengths


def load_model_for_visualization(
    model_path: str,
    model_type: Optional[str] = None,
    bandconfig: Optional[str] = None,
) -> nn.Module:
    """
    Load a model for visualization purposes (returns model without running inference).
    
    Args:
        model_path: Path to model checkpoint or HuggingFace model name
        model_type: Model type ('hf_pretrained', 'checkpoint_bigearthnet', 'checkpoint_multimodal')
        bandconfig: Band configuration
    
    Returns:
        Loaded model in eval mode
    """
    if model_type is None:
        model_type = _detect_model_type(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "hf_pretrained":
        # Load from HuggingFace
        from huggingface_hub import snapshot_download
        import tempfile
        
        print(f"Loading HuggingFace model: {model_path}")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(snapshot_download(repo_id=model_path, cache_dir=tmpdir))
            
            config_file = local_path / "config.json"
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Convert dict to ILMConfiguration (handle both flat and nested structures)
            if isinstance(config_dict, dict):
                # Try to extract from nested structure first
                if 'config' in config_dict and isinstance(config_dict['config'], dict):
                    config_data = config_dict['config']
                else:
                    config_data = config_dict
                
                # Extract ILMConfiguration parameters with defaults
                network_type_str = config_data.get('network_type', 'IMAGE_CLASSIFICATION')
                if isinstance(network_type_str, str):
                    network_type = ILMType.IMAGE_CLASSIFICATION  # Default
                else:
                    network_type = network_type_str
                
                config = ILMConfiguration(
                    network_type=network_type,
                    classes=config_data.get('classes', 19),
                    image_size=config_data.get('image_size', 120),
                    drop_rate=config_data.get('drop_rate', 0.15),
                    drop_path_rate=config_data.get('drop_path_rate', 0.15),
                    timm_model_name=config_data.get('timm_model_name', 
                                                    config_data.get('architecture', 
                                                                  config_data.get('timm_model_name', 'resnet101'))),
                    channels=config_data.get('channels', 10),
                )
            else:
                raise ValueError(f"Unexpected config format: {type(config_dict)}")
            
            model_file = local_path / "pytorch_model.bin"
            if not model_file.exists():
                model_file = local_path / "model.safetensors"
            
            if not model_file.exists():
                raise FileNotFoundError(f"Model weights not found in {local_path}")
            
            model = BigEarthNetv2_0_ImageClassifier(
                config=config,
                lr=config_dict.get('lr', 0.001),
                warmup=config_dict.get('warmup', None),
                dinov3_model_name=config_dict.get('dinov3_model_name', None),
                linear_probe=config_dict.get('linear_probe', False),
                head_type=config_dict.get('head_type', 'linear'),
                mlp_hidden_dims=config_dict.get('mlp_hidden_dims', None),
                head_dropout=config_dict.get('head_dropout', None),
            )
            
            if model_file.suffix == '.safetensors':
                from safetensors.torch import load_file
                state_dict = load_file(str(model_file))
            else:
                state_dict = torch.load(model_file, map_location='cpu')
            
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            model.load_state_dict(state_dict, strict=False)
    
    elif model_type == "checkpoint_bigearthnet":
        # Load BigEarthNet checkpoint
        ckpt_path = Path(model_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        hparams = checkpoint.get('hyper_parameters', {})
        config_data = hparams.get('config', {})
        
        # Create ILMConfiguration from config data
        if isinstance(config_data, dict):
            network_type_str = config_data.get('network_type', 'IMAGE_CLASSIFICATION')
            if isinstance(network_type_str, str):
                network_type = ILMType.IMAGE_CLASSIFICATION
            else:
                network_type = network_type_str
            
            config = ILMConfiguration(
                network_type=network_type,
                classes=config_data.get('classes', 19),
                image_size=config_data.get('image_size', 120),
                drop_rate=config_data.get('drop_rate', 0.15),
                drop_path_rate=config_data.get('drop_path_rate', 0.15),
                timm_model_name=config_data.get('timm_model_name', 
                                                config_data.get('architecture', 'resnet101')),
                channels=config_data.get('channels', 10),
            )
        else:
            # Fallback: use hparams directly if config is not a dict
            config = ILMConfiguration(
                network_type=ILMType.IMAGE_CLASSIFICATION,
                classes=19,
                image_size=120,
                drop_rate=hparams.get('dropout', 0.15),
                drop_path_rate=hparams.get('drop_path_rate', 0.15),
                timm_model_name=hparams.get('architecture', 'resnet101'),
                channels=hparams.get('channels', 10),
            )
        
        model = BigEarthNetv2_0_ImageClassifier.load_from_checkpoint(
            str(ckpt_path),
            config=config,
            lr=hparams.get('lr', 0.001),
            warmup=hparams.get('warmup', None),
        )
    
    elif model_type == "checkpoint_multimodal":
        # Load multimodal checkpoint
        ckpt_path = Path(model_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        hparams = checkpoint.get('hyper_parameters', {})
        dinov3_model_name = _infer_dinov3_model_from_checkpoint(model_path)
        
        # Infer configuration from checkpoint
        use_s1 = hparams.get('use_s1', True)
        fusion_type = hparams.get('fusion_type', 'concat')
        
        s2_bands = STANDARD_BANDS.get("S2", STANDARD_BANDS.get("s2_full", []))
        s1_bands = STANDARD_BANDS.get("S1", ["VV", "VH"])
        rgb_bands = ["B04", "B03", "B02"]
        full_s2_non_rgb = [b for b in s2_bands if b not in rgb_bands]
        s2_non_rgb = full_s2_non_rgb
        s2_ordered = rgb_bands + s2_non_rgb
        resnet_input_channels = len(s2_non_rgb) + (len(s1_bands) if use_s1 else 0)
        use_resnet = resnet_input_channels > 0
        
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
            "fusion": {"type": fusion_type},
            "classifier": {
                "type": hparams.get('classifier_type', 'linear'),
                "num_classes": 19,
                "drop_rate": hparams.get('dropout', 0.15),
            },
            "image_size": 120,
        }
        
        model = MultiModalLightningModule.load_from_checkpoint(
            str(ckpt_path),
            config=config,
            lr=hparams.get('lr', 0.001),
            warmup=hparams.get('warmup', None),
            dinov3_checkpoint=None,
            resnet_checkpoint=None,
            freeze_dinov3=False,
            freeze_resnet=False,
            dinov3_model_name=dinov3_model_name,
            threshold=0.5,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    model = model.to(device)
    return model


def extract_resnet_backbone(model: nn.Module) -> Optional[nn.Module]:
    """
    Extract ResNet backbone from a model.
    
    Args:
        model: The model to extract backbone from
    
    Returns:
        ResNet backbone module or None if not found
    """
    # Try different model structures
    if hasattr(model, 'model'):
        # MultiModalLightningModule structure
        if hasattr(model.model, 'resnet_backbone'):
            return model.model.resnet_backbone
        # BigEarthNetv2_0_ImageClassifier with ResNet
        if hasattr(model.model, 'backbone'):
            backbone = model.model.backbone
            # Check if it's a ResNet
            if 'resnet' in str(type(backbone)).lower():
                return backbone
    
    # Try direct access
    if hasattr(model, 'resnet_backbone'):
        return model.resnet_backbone
    
    # Try to find ResNet in children
    for name, module in model.named_modules():
        if 'resnet' in name.lower() and 'backbone' in name.lower():
            return module
    
    return None


def find_target_layer_resnet(resnet_backbone: nn.Module) -> Optional[nn.Module]:
    """
    Find the target layer for Grad-CAM in ResNet (typically the last convolutional layer).
    
    Args:
        resnet_backbone: ResNet backbone module
    
    Returns:
        Target layer module or None if not found
    """
    # ResNet structure: layer4 is typically the last conv block
    if hasattr(resnet_backbone, 'layer4'):
        return resnet_backbone.layer4[-1]  # Last block in layer4
    
    # Alternative: look for the last conv layer
    for name, module in list(resnet_backbone.named_modules())[::-1]:
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            # Find the parent module (the block containing this conv)
            parts = name.split('.')
            if len(parts) >= 2:
                parent_name = '.'.join(parts[:-1])
                parent = dict(resnet_backbone.named_modules())[parent_name]
                return parent
    
    return None


class GradCAM:
    """Grad-CAM implementation for ResNet models."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not captured.")
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam


def find_success_cases(
    baseline_probs: np.ndarray,
    your_probs: np.ndarray,
    labels: np.ndarray,
    target_classes: List[int],
    threshold: float = 0.5,
    min_confidence_diff: float = 0.3,
    n_samples: int = 5,
) -> List[Tuple[int, int]]:
    """
    Find success cases where your model correctly predicts target classes
    that the baseline misses.
    
    Returns:
        List of (sample_idx, class_idx) tuples
    """
    success_cases = []
    
    for idx in range(len(labels)):
        has_target = any(labels[idx, cls_idx] > 0.5 for cls_idx in target_classes)
        if not has_target:
            continue
        
        for cls_idx in target_classes:
            if labels[idx, cls_idx] > 0.5:
                baseline_pred = baseline_probs[idx, cls_idx] >= threshold
                your_pred = your_probs[idx, cls_idx] >= threshold
                
                if not baseline_pred and your_pred:
                    confidence_diff = your_probs[idx, cls_idx] - baseline_probs[idx, cls_idx]
                    if confidence_diff >= min_confidence_diff:
                        success_cases.append((idx, cls_idx, confidence_diff))
                        break
    
    success_cases.sort(key=lambda x: x[2], reverse=True)
    return [(idx, cls_idx) for idx, cls_idx, _ in success_cases[:n_samples]]


def find_failure_cases(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    target_class_idx: int,
    threshold: float = 0.5,
    n_samples: int = 5,
) -> List[Tuple[int, int]]:
    """
    Find failure cases where model predicts wrong class for target class.
    
    Returns:
        List of (sample_idx, predicted_class_idx) tuples
    """
    failure_cases = []
    
    for idx in range(len(labels)):
        if labels[idx, target_class_idx] <= 0.5:
            continue
        
        if predictions[idx, target_class_idx] > 0.5:
            continue  # Correct prediction
        
        predicted_classes = np.where(predictions[idx] > 0.5)[0]
        if len(predicted_classes) == 0:
            predicted_class = np.argmax(probabilities[idx])
        else:
            predicted_class = predicted_classes[np.argmax(probabilities[idx, predicted_classes])]
        
        if predicted_class == target_class_idx:
            continue
        
        failure_cases.append((idx, predicted_class))
    
    failure_cases.sort(
        key=lambda x: probabilities[x[0], x[1]],
        reverse=True
    )
    
    return failure_cases[:n_samples]


def extract_spectral_profile(
    image_tensor: torch.Tensor,
    band_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract spectral profile (mean intensity per band) from image."""
    intensities = image_tensor[0].mean(dim=(1, 2)).cpu().numpy()
    
    if intensities.max() > intensities.min():
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
    
    wavelengths = get_band_wavelengths(band_names)
    return np.array(wavelengths), intensities


def visualize_spectral_profile(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    band_names: List[str],
    ground_truth_class: str,
    predicted_class: str,
    sample_idx: int,
    output_path: str,
    figsize: Tuple[int, int] = (14, 8),
) -> None:
    """Visualize spectral profile with annotations."""
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(wavelengths, intensities, 'o-', linewidth=2, markersize=8, color='blue', label='Spectral Profile')
    
    for i, (w, intensity, band) in enumerate(zip(wavelengths, intensities, band_names)):
        ax.annotate(
            band,
            (w, intensity),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.7,
        )
    
    rgb_indices = [i for i, band in enumerate(band_names) if band in ["B02", "B03", "B04"]]
    if rgb_indices:
        rgb_wavelengths = wavelengths[rgb_indices]
        ax.fill_between(
            [rgb_wavelengths.min() - 50, rgb_wavelengths.max() + 50],
            [0, 0],
            [1, 1],
            alpha=0.2,
            color='red',
            label='RGB Region'
        )
    
    nir_rededge_indices = [i for i, band in enumerate(band_names) 
                           if band in ["B05", "B06", "B07", "B08", "B8A"]]
    if nir_rededge_indices:
        nir_wavelengths = wavelengths[nir_rededge_indices]
        nir_intensities = intensities[nir_rededge_indices]
        ax.fill_between(
            [nir_wavelengths.min() - 50, nir_wavelengths.max() + 50],
            [0, 0],
            [1, 1],
            alpha=0.2,
            color='green',
            label='NIR/Red-Edge Region'
        )
        
        if len(nir_intensities) > 0:
            max_nir_idx = np.argmax(nir_intensities)
            max_nir_w = nir_wavelengths[max_nir_idx]
            max_nir_int = nir_intensities[max_nir_idx]
            ax.annotate(
                'Red-Edge Spike\n(Distinguishing feature)',
                (max_nir_w, max_nir_int),
                xytext=(20, 20),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
            )
    
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Spectral Profile - Failure Case #{sample_idx + 1}\n'
        f'Ground Truth: {ground_truth_class} | Predicted: {predicted_class}\n'
        f'RGB channels (left) are similar, but NIR/Red-Edge (right) differ',
        fontsize=13,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim([wavelengths.min() - 100, wavelengths.max() + 100])
    ax.set_ylim([-0.05, 1.05])
    
    if len(rgb_indices) > 0 and len(nir_rededge_indices) > 0:
        separation = (wavelengths[rgb_indices].max() + wavelengths[nir_rededge_indices].min()) / 2
        ax.axvline(separation, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved spectral profile to: {output_path}")
    plt.close()


def visualize_comparison(
    image: np.ndarray,
    dinov3_attention: np.ndarray,
    resnet_gradcam: np.ndarray,
    class_name: str,
    sample_idx: int,
    output_path: str,
    figsize: Tuple[int, int] = (20, 5),
) -> None:
    """Visualize side-by-side comparison of DINOv3 attention and ResNet Grad-CAM."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(image)
    im1 = axes[1].imshow(dinov3_attention, cmap='hot', alpha=0.6, interpolation='bilinear')
    axes[1].set_title("DINOv3 CLS Attention\n(Tightly contours objects)", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].imshow(image)
    im2 = axes[2].imshow(resnet_gradcam, cmap='hot', alpha=0.6, interpolation='bilinear')
    axes[2].set_title("ResNet Grad-CAM\n(May show scattered noise)", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    axes[3].imshow(dinov3_attention, cmap='hot', interpolation='bilinear')
    axes[3].set_title("DINOv3 Attention\n(Standalone)", fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.suptitle(
        f"Success Case #{sample_idx + 1}: {class_name}\n"
        f"Baseline missed it, Your model detected it",
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    plt.close()


def main(
    mode: str = typer.Option(..., help="Mode: 'win' for success cases or 'lost' for failure cases"),
    model_path: str = typer.Option(..., help="Path to your model"),
    baseline_model_path: Optional[str] = typer.Option(None, help="Path to baseline model (required for 'win' mode)"),
    target_labels: str = typer.Option(..., help="Comma-separated target class names (e.g., 'Industrial or commercial units,Coastal wetlands' or 'Permanent crops')"),
    model_type: Optional[str] = typer.Option(None, help="Your model type"),
    baseline_model_type: Optional[str] = typer.Option(None, help="Baseline model type"),
    threshold: float = typer.Option(0.5, help="Classification threshold"),
    min_confidence_diff: float = typer.Option(0.3, help="Minimum confidence difference (for 'win' mode)"),
    n_samples: int = typer.Option(5, help="Number of cases to visualize"),
    seed: int = typer.Option(42, help="Random seed"),
    bs: int = typer.Option(32, help="Batch size"),
    workers: int = typer.Option(8, help="Number of workers"),
    bandconfig: str = typer.Option("all", help="Band configuration"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory"),
):
    """
    Visualize success or failure cases for model analysis.
    
    Mode 'win': Compare your model vs baseline for success cases
    Mode 'lost': Analyze failure cases with spectral profiles
    """
    if mode not in ['win', 'lost']:
        raise ValueError(f"Mode must be 'win' or 'lost', got '{mode}'")
    
    if mode == 'win' and baseline_model_path is None:
        raise ValueError("baseline_model_path is required for 'win' mode")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Parse target labels
    target_classes_info = parse_class_names(target_labels)
    target_class_indices = [idx for idx, _ in target_classes_info]
    target_class_names = [name for _, name in target_classes_info]
    
    if output_dir is None:
        mode_name = "why_we_won" if mode == 'win' else "why_we_lost"
        output_dir = Path(".") / f"{mode_name}_visualizations"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"{'Why We Won' if mode == 'win' else 'Why We Lost'} Visualization")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Target classes: {', '.join(target_class_names)}")
    print(f"Your model: {model_path}")
    if mode == 'win':
        print(f"Baseline model: {baseline_model_path}")
    print("=" * 80 + "\n")
    
    # Load model and run inference
    print("Loading your model and running inference...")
    predictions, probabilities, labels, _ = load_model_and_infer(
        model_path=model_path,
        model_type=model_type,
        bs=bs,
        workers=workers,
        threshold=threshold,
        bandconfig=bandconfig,
    )
    
    if mode == 'win':
        # Load baseline model
        print("Loading baseline model and running inference...")
        _, baseline_probs, _, _ = load_model_and_infer(
            model_path=baseline_model_path,
            model_type=baseline_model_type,
            bs=bs,
            workers=workers,
            threshold=threshold,
            bandconfig=bandconfig,
        )
        
        # Find success cases
        print("\nFinding success cases...")
        success_cases = find_success_cases(
            baseline_probs,
            probabilities,
            labels,
            target_class_indices,
            threshold=threshold,
            min_confidence_diff=min_confidence_diff,
            n_samples=n_samples,
        )
        
        if len(success_cases) == 0:
            print("⚠ No success cases found. Try lowering min_confidence_diff or threshold.")
            return
        
        print(f"✓ Found {len(success_cases)} success cases")
        
        # Load models for visualization
        print("\nLoading models for visualization...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        your_model = load_model_for_visualization(model_path, model_type, bandconfig)
        baseline_model = load_model_for_visualization(baseline_model_path, baseline_model_type, bandconfig)
        
        # Extract backbones
        print("Extracting backbones...")
        dinov3_backbone = extract_dinov3_backbone(your_model)
        resnet_backbone = extract_resnet_backbone(baseline_model)
        
        if dinov3_backbone is None:
            print("⚠ Warning: Could not extract DINOv3 backbone from your model. Skipping attention visualization.")
            dinov3_backbone = None
        
        if resnet_backbone is None:
            print("⚠ Warning: Could not extract ResNet backbone from baseline model. Skipping Grad-CAM visualization.")
            resnet_backbone = None
        
        # Setup dataloader to get actual images
        hostname, data_dirs = get_benv2_dir_dict()
        data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
        
        bands, num_channels = get_bands(bandconfig)
        
        hparams = {
            "batch_size": 1,
            "workers": 1,
            "channels": num_channels,
            "bandconfig": bandconfig,
        }
        dm = default_dm(hparams, data_dirs, 120)
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()
        
        # Find target layer for Grad-CAM
        target_layer = None
        gradcam = None
        if resnet_backbone is not None:
            target_layer = find_target_layer_resnet(resnet_backbone)
            if target_layer is None:
                print("⚠ Warning: Could not find target layer for Grad-CAM. Skipping Grad-CAM visualization.")
            else:
                # GradCAM needs the full model, not just the backbone
                gradcam = GradCAM(baseline_model, target_layer)
        
        # Visualize each success case
        for vis_idx, (sample_idx, class_idx) in enumerate(success_cases):
            print(f"\nProcessing success case {vis_idx + 1}/{len(success_cases)} (sample {sample_idx})...")
            
            # Get the image
            for idx, (x, y) in enumerate(test_loader):
                if idx == sample_idx:
                    image_tensor = x
                    image_labels = y
                    break
            
            # Extract RGB for visualization
            rgb_image = image_tensor[0, :3, :, :].permute(1, 2, 0).numpy()
            rgb_image = np.clip(rgb_image, 0, 1)
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            else:
                rgb_image = rgb_image.astype(np.uint8)
            
            class_name = NEW_LABELS[class_idx]
            print(f"  Class: {class_name}")
            print(f"  Baseline prob: {baseline_probs[sample_idx, class_idx]:.3f}")
            print(f"  Your model prob: {probabilities[sample_idx, class_idx]:.3f}")
            
            # Extract DINOv3 attention
            dinov3_attention = None
            if dinov3_backbone is not None:
                try:
                    # Set attention implementation to 'eager' to support output_attentions
                    if hasattr(dinov3_backbone, 'set_attn_implementation'):
                        try:
                            dinov3_backbone.set_attn_implementation('eager')
                        except Exception:
                            pass  # Some models don't support this
                    
                    # Prepare RGB input for DINOv3 (first 3 channels)
                    rgb_input = image_tensor[:, :3, :, :].to(device)
                    
                    # Get attention
                    _, attention_weights, patch_h, patch_w = get_patch_embeddings_and_attention(
                        dinov3_backbone,
                        rgb_input,
                        return_attention=True,
                    )
                    
                    if attention_weights is not None:
                        # Extract CLS token attention
                        # attention_weights shape: (B, N_heads, N_tokens, N_tokens) or (N_heads, N_tokens, N_tokens)
                        if len(attention_weights.shape) == 4:
                            attn = attention_weights[0].cpu().numpy()  # (N_heads, N_tokens, N_tokens)
                        else:
                            attn = attention_weights.cpu().numpy()  # (N_heads, N_tokens, N_tokens)
                        
                        cls_attn = attn[:, 0, :].mean(axis=0)  # Average across heads, shape: (N_tokens,)
                        
                        # Get patch tokens (skip CLS and register tokens)
                        n_tokens = len(cls_attn)
                        expected_n_patches = patch_h * patch_w
                        
                        # Calculate actual number of patch tokens
                        # DINOv3 typically has: 1 CLS + 4 register tokens = 5 special tokens
                        # But some variants may differ
                        n_special_tokens = n_tokens - expected_n_patches
                        
                        if n_special_tokens == 5:
                            # Standard DINOv3: 1 CLS + 4 registers
                            patch_start_idx = 5
                            actual_n_patches = expected_n_patches
                        elif n_special_tokens == 1:
                            # Just CLS token
                            patch_start_idx = 1
                            actual_n_patches = expected_n_patches
                        elif n_special_tokens > 0 and n_special_tokens <= 10:
                            # Try to infer
                            patch_start_idx = n_special_tokens
                            actual_n_patches = expected_n_patches
                        else:
                            # Fallback: recalculate patch dimensions from actual token count
                            # Assume patches form a square or near-square grid
                            actual_n_patches = n_tokens - max(1, n_special_tokens)
                            patch_h = int(np.sqrt(actual_n_patches))
                            patch_w = actual_n_patches // patch_h
                            if patch_h * patch_w != actual_n_patches:
                                # Not a perfect square, adjust
                                patch_w = int(np.ceil(np.sqrt(actual_n_patches)))
                                patch_h = (actual_n_patches + patch_w - 1) // patch_w
                            patch_start_idx = n_tokens - actual_n_patches
                        
                        # Extract patch attention tokens
                        if patch_start_idx + actual_n_patches <= n_tokens:
                            patch_attn = cls_attn[patch_start_idx:patch_start_idx + actual_n_patches]
                        else:
                            # Fallback: use all tokens except first few
                            patch_attn = cls_attn[patch_start_idx:]
                            actual_n_patches = len(patch_attn)
                            # Recalculate grid
                            patch_h = int(np.sqrt(actual_n_patches))
                            patch_w = actual_n_patches // patch_h
                            if patch_h * patch_w != actual_n_patches:
                                patch_w = int(np.ceil(np.sqrt(actual_n_patches)))
                                patch_h = (actual_n_patches + patch_w - 1) // patch_w
                        
                        # Reshape to spatial grid
                        if len(patch_attn) == patch_h * patch_w:
                            patch_attn = patch_attn.reshape(patch_h, patch_w)
                        else:
                            # Pad or truncate to fit
                            target_size = patch_h * patch_w
                            if len(patch_attn) < target_size:
                                # Pad with zeros
                                pad_size = target_size - len(patch_attn)
                                patch_attn = np.pad(patch_attn, (0, pad_size), mode='constant', constant_values=0)
                            else:
                                # Truncate
                                patch_attn = patch_attn[:target_size]
                            patch_attn = patch_attn.reshape(patch_h, patch_w)
                        
                        # Upsample to image size
                        img_h, img_w = rgb_image.shape[:2]
                        zoom_factors = (img_h / patch_h, img_w / patch_w)
                        dinov3_attention = zoom(patch_attn, zoom_factors, order=1)
                        dinov3_attention = (dinov3_attention - dinov3_attention.min()) / (dinov3_attention.max() - dinov3_attention.min() + 1e-8)
                        print(f"  ✓ Extracted DINOv3 attention (patches: {patch_h}x{patch_w}, tokens: {n_tokens})")
                except Exception as e:
                    import traceback
                    print(f"  ⚠ Failed to extract DINOv3 attention: {e}")
                    print(f"  Traceback: {traceback.format_exc()}")
                    dinov3_attention = None
            
            # Extract ResNet Grad-CAM
            resnet_gradcam = None
            if gradcam is not None:
                try:
                    # Prepare input (use all channels or just RGB depending on model)
                    if image_tensor.shape[1] == 3:
                        model_input = image_tensor.to(device)
                    else:
                        # Use first 3 channels (RGB) for ResNet if it expects RGB
                        model_input = image_tensor[:, :3, :, :].to(device)
                    
                    resnet_gradcam = gradcam.generate_cam(model_input, class_idx)
                    print(f"  ✓ Extracted ResNet Grad-CAM")
                except Exception as e:
                    print(f"  ⚠ Failed to extract ResNet Grad-CAM: {e}")
                    resnet_gradcam = None
            
            # Create visualization
            if dinov3_attention is not None or resnet_gradcam is not None:
                # Use placeholder if one is missing
                if dinov3_attention is None:
                    dinov3_attention = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
                if resnet_gradcam is None:
                    resnet_gradcam = np.zeros((rgb_image.shape[0], rgb_image.shape[1]))
                
                output_path = output_dir / f"success_case_{vis_idx + 1}_{class_name.replace(' ', '_')}.png"
                visualize_comparison(
                    rgb_image,
                    dinov3_attention,
                    resnet_gradcam,
                    class_name,
                    vis_idx,
                    str(output_path),
                )
            else:
                print(f"  ⚠ Skipping visualization - could not extract attention maps")
    
    else:  # mode == 'lost'
        if len(target_class_indices) != 1:
            raise ValueError("'lost' mode requires exactly one target class")
        
        target_class_idx = target_class_indices[0]
        target_class_name = target_class_names[0]
        
        # Find failure cases
        print("\nFinding failure cases...")
        failure_cases = find_failure_cases(
            predictions,
            probabilities,
            labels,
            target_class_idx,
            threshold=threshold,
            n_samples=n_samples,
        )
        
        if len(failure_cases) == 0:
            print(f"⚠ No failure cases found. The model may be performing well on {target_class_name}.")
            return
        
        print(f"✓ Found {len(failure_cases)} failure cases")
        
        # Setup dataloader to get actual images with all bands
        hostname, data_dirs = get_benv2_dir_dict()
        data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
        
        bands, num_channels = get_bands(bandconfig)
        
        hparams = {
            "batch_size": 1,
            "workers": 1,
            "channels": num_channels,
            "bandconfig": bandconfig,
        }
        dm = default_dm(hparams, data_dirs, 120)
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()
        
        # Visualize each failure case
        for vis_idx, (sample_idx, predicted_class_idx) in enumerate(failure_cases):
            print(f"\nProcessing failure case {vis_idx + 1}/{len(failure_cases)} (sample {sample_idx})...")
            
            # Get the image
            for idx, (x, y) in enumerate(test_loader):
                if idx == sample_idx:
                    image_tensor = x
                    break
            
            # Extract spectral profile
            wavelengths, intensities = extract_spectral_profile(image_tensor, bands)
            
            predicted_class = NEW_LABELS[predicted_class_idx]
            
            print(f"  Ground Truth: {target_class_name}")
            print(f"  Predicted: {predicted_class}")
            print(f"  Confidence: {probabilities[sample_idx, predicted_class_idx]:.3f}")
            
            # Visualize
            output_path = output_dir / f"failure_case_{vis_idx + 1}_{predicted_class.replace(' ', '_')}.png"
            visualize_spectral_profile(
                wavelengths,
                intensities,
                bands,
                target_class_name,
                predicted_class,
                vis_idx,
                str(output_path),
            )
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    typer.run(main)

