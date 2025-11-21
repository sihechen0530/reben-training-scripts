"""
Utility helpers for loading BigEarthNet models for diagnosis scripts.

Supports three model types:
1. HuggingFace pretrained models (model_name string)
2. Local BigEarthNet checkpoints (.ckpt file)
3. Local Multimodal checkpoints (.ckpt file)
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
from configilm.extra.BENv2_utils import resolve_data_dir

import numpy as np
import lightning.pytorch as pl
import torch

# Ensure project root on sys.path
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configilm.ConfigILM import ILMConfiguration, ILMType  # noqa: E402
from configilm.extra.BENv2_utils import STANDARD_BANDS  # noqa: E402
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet  # noqa: E402

from reben_publication.BigEarthNetv2_0_ImageClassifier import (  # noqa: E402
    BigEarthNetv2_0_ImageClassifier,
)
from multimodal.lightning_module import MultiModalLightningModule  # noqa: E402
from scripts.utils import get_benv2_dir_dict, get_bands, default_dm  # noqa: E402


def _infer_dinov3_model_from_checkpoint(checkpoint_path: str) -> str:
    """Infer DINOv3 model name from checkpoint."""
    ckpt_path = Path(checkpoint_path)
    filename = ckpt_path.name.lower()

    if "dinov3-large" in filename or "dinov3-l" in filename:
        return "facebook/dinov3-vitl16-pretrain-lvd1689m"
    if "dinov3-base" in filename or "dinov3-b" in filename:
        return "facebook/dinov3-vitb16-pretrain-lvd1689m"
    if "dinov3-small" in filename or "dinov3-s" in filename:
        return "facebook/dinov3-vits16-pretrain-lvd1689m"
    if "dinov3-giant" in filename or "dinov3-g" in filename:
        return "facebook/dinov3-vitg16-pretrain-lvd1689m"

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            if "dinov3_model_name" in hparams:
                return hparams["dinov3_model_name"]
    except Exception:
        pass

    return "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _detect_model_type(model_path: str) -> str:
    """Detect model type: hf_pretrained, checkpoint_bigearthnet, or checkpoint_multimodal."""
    if (
        "/" in model_path
        and not model_path.endswith(".ckpt")
        and not Path(model_path).exists()
    ):
        return "hf_pretrained"

    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        return "hf_pretrained"

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", {})
        hparams = checkpoint.get("hyper_parameters", {})

        multimodal_keys = [
            "model.dinov3_backbone",
            "model.resnet_backbone",
            "model.fusion",
        ]
        has_multimodal_keys = any(
            key in str(state_dict.keys()) for key in multimodal_keys
        )

        if "use_s1" in hparams or "fusion_type" in hparams:
            return "checkpoint_multimodal"
        if has_multimodal_keys:
            return "checkpoint_multimodal"
        return "checkpoint_bigearthnet"
    except Exception:
        return "checkpoint_bigearthnet"


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
    allow_mock_data: bool = True,
    return_sample_ids: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load model (HF pretrained, BigEarthNet checkpoint, or Multimodal checkpoint) and run inference.

    Returns:
        predictions, probabilities, labels, optional sample_ids
    """
    assert (
        Path(".").resolve().name == "scripts"
    ), "Please run this script from the scripts directory."

    if model_type is None:
        model_type = _detect_model_type(model_path)

    print(f"\n{'=' * 60}")
    print(f"Model Type: {model_type}")
    print(f"Model Path: {model_path}")
    print(f"{'=' * 60}\n")

    num_classes = 19
    img_size = 120

    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # HuggingFace pretrained --------------------------------------------------
    if model_type == "hf_pretrained":
        print("Loading HuggingFace pretrained model...")
        model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_path)
        model.eval()

        channels = model.config.channels
        architecture = getattr(model.config, "timm_model_name", "unknown")
        channel_to_bandconfig = {
            12: "all",
            10: "s2",
            2: "s1",
            13: "all_full",
            11: "s2_full",
            3: "rgb",
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
        data_dirs = resolve_data_dir(data_dirs, allow_mock=allow_mock_data)
        dm = default_dm(hparams, data_dirs, img_size)

    # BigEarthNet checkpoint ---------------------------------------------------
    elif model_type == "checkpoint_bigearthnet":
        print("Loading BigEarthNet checkpoint...")
        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        checkpoint_channels = None
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if "hyper_parameters" in checkpoint:
            hparams_ckpt = checkpoint["hyper_parameters"]
            architecture = architecture or hparams_ckpt.get(
                "architecture", architecture
            )
            bandconfig = bandconfig or hparams_ckpt.get("bandconfig", bandconfig)
            checkpoint_channels = hparams_ckpt.get("channels", checkpoint_channels)
            if drop_rate == 0.15 and "dropout" in hparams_ckpt:
                drop_rate = hparams_ckpt["dropout"]
            if drop_path_rate == 0.15 and "drop_path_rate" in hparams_ckpt:
                drop_path_rate = hparams_ckpt["drop_path_rate"]
            if seed == 42 and "seed" in hparams_ckpt:
                seed = hparams_ckpt["seed"]

        if architecture is None or bandconfig is None:
            filename_parts = ckpt_path.stem.split("-")
            if len(filename_parts) >= 3 and architecture is None:
                potential_arch = filename_parts[0]
                if potential_arch and not potential_arch.isdigit():
                    architecture = potential_arch

            if bandconfig is None:
                channels = checkpoint_channels
                if channels is None:
                    try:
                        channels = int(filename_parts[3])
                    except (ValueError, IndexError):
                        channels = None

                if channels is not None:
                    channel_to_bandconfig = {
                        12: "all",
                        10: "s2",
                        2: "s1",
                        13: "all_full",
                        11: "s2_full",
                        3: "rgb",
                    }
                    bandconfig = channel_to_bandconfig.get(channels, "s2")

        if architecture is None or bandconfig is None:
            raise ValueError(
                "Could not infer architecture/bandconfig. Please provide them explicitly."
            )

        _, inferred_channels = get_bands(bandconfig)
        if checkpoint_channels is not None:
            if inferred_channels != checkpoint_channels:
                print(
                    f"Warning: bandconfig '{bandconfig}' implies {inferred_channels} channels "
                    f"but checkpoint was trained with {checkpoint_channels} channels. "
                    "Proceeding with checkpoint channel count."
                )
            channels = checkpoint_channels
        else:
            channels = inferred_channels

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
        if architecture.startswith("dinov3") and dinov3_name is None:
            arch_lower = architecture.lower()
            if "small" in arch_lower or arch_lower.endswith("s"):
                dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
            elif "base" in arch_lower or arch_lower.endswith("b"):
                dinov3_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
            elif "large" in arch_lower or arch_lower.endswith("l"):
                dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
            elif "giant" in arch_lower or arch_lower.endswith("g"):
                dinov3_name = "facebook/dinov3-vitg16-pretrain-lvd1689m"
            else:
                dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"

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

        # Extract class weights from checkpoint if available
        class_weights = None
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Check if loss.pos_weight exists in checkpoint
            if 'loss.pos_weight' in state_dict:
                class_weights = state_dict['loss.pos_weight']
                print(f"Loaded class weights from checkpoint (shape: {class_weights.shape})")
                print(f"  Min weight: {class_weights.min():.4f}")
                print(f"  Max weight: {class_weights.max():.4f}")
        elif 'loss.pos_weight' in checkpoint:
            class_weights = checkpoint['loss.pos_weight']
            print(f"Loaded class weights from checkpoint (shape: {class_weights.shape})")
            print(f"  Min weight: {class_weights.min():.4f}")
            print(f"  Max weight: {class_weights.max():.4f}")

        model = BigEarthNetv2_0_ImageClassifier.load_from_checkpoint(
            str(ckpt_path),
            config=config,
            lr=lr,
            warmup=warmup,
            dinov3_model_name=dinov3_name,
            class_weights=class_weights,  # Pass class weights to match training
        )
        model.eval()

    # Multimodal checkpoint ----------------------------------------------------
    elif model_type == "checkpoint_multimodal":
        print("Loading Multimodal checkpoint...")
        ckpt_path = Path(model_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        checkpoint_channels = None
        checkpoint_use_s1 = None
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        if "hyper_parameters" in checkpoint:
            hparams_ckpt = checkpoint["hyper_parameters"]
            architecture = architecture or hparams_ckpt.get(
                "architecture", architecture
            )
            bandconfig = bandconfig or hparams_ckpt.get("bandconfig", bandconfig)
            if use_s1 is None and "use_s1" in hparams_ckpt:
                checkpoint_use_s1 = hparams_ckpt["use_s1"]
                use_s1 = checkpoint_use_s1
            checkpoint_channels = hparams_ckpt.get("channels", checkpoint_channels)
            if drop_rate == 0.15 and "dropout" in hparams_ckpt:
                drop_rate = hparams_ckpt["dropout"]
            if seed == 42 and "seed" in hparams_ckpt:
                seed = hparams_ckpt["seed"]

        if dinov3_model_name is None:
            if architecture is not None:
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
            "rgb": "rgb",
            "truecolor": "rgb",
            "s2": "s2",
            "s2_only": "s2",
            "s2s1": "s2s1",
            "all": "s2s1",
            "multimodal": "s2s1",
        }
        effective_bandconfig = band_map.get(band_alias, "s2s1")

        if use_s1 is None:
            use_s1 = effective_bandconfig == "s2s1"
            if checkpoint_use_s1 is not None:
                use_s1 = checkpoint_use_s1

        s2_bands = STANDARD_BANDS.get("S2", STANDARD_BANDS.get("s2_full", []))
        s1_bands = STANDARD_BANDS.get("S1", ["VV", "VH"])
        rgb_bands = ["B04", "B03", "B02"]
        full_s2_non_rgb = [b for b in s2_bands if b not in rgb_bands]
        s2_non_rgb = [] if effective_bandconfig == "rgb" else full_s2_non_rgb
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
            if "hyper_parameters" in checkpoint:
                hparams = checkpoint["hyper_parameters"]
                if "fusion_type" in hparams:
                    config["fusion"]["type"] = hparams["fusion_type"]
                if "classifier_type" in hparams:
                    config["classifier"]["type"] = hparams["classifier_type"]
                if (
                    "classifier_hidden_dim" in hparams
                    and config["classifier"]["type"] == "mlp"
                ):
                    config["classifier"]["hidden_dim"] = hparams[
                        "classifier_hidden_dim"
                    ]
        except Exception:
            pass

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

        # Extract class weights from checkpoint if available
        class_weights = None
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Check if loss.pos_weight exists in checkpoint
            if 'loss.pos_weight' in state_dict:
                class_weights = state_dict['loss.pos_weight']
                print(f"Loaded class weights from checkpoint (shape: {class_weights.shape})")
                print(f"  Min weight: {class_weights.min():.4f}")
                print(f"  Max weight: {class_weights.max():.4f}")
        elif 'loss.pos_weight' in checkpoint:
            class_weights = checkpoint['loss.pos_weight']
            print(f"Loaded class weights from checkpoint (shape: {class_weights.shape})")
            print(f"  Min weight: {class_weights.min():.4f}")
            print(f"  Max weight: {class_weights.max():.4f}")

        # Load checkpoint with handling for architecture mismatches
        try:
            model = MultiModalLightningModule.load_from_checkpoint(
                str(ckpt_path),
                config=config,
                lr=lr,
                warmup=None if warmup == -1 else warmup,
                dinov3_checkpoint=None,
                resnet_checkpoint=None,
                freeze_dinov3=False,
                freeze_resnet=False,
                dinov3_model_name=dinov3_model_name,
                class_weights=class_weights,  # Pass class weights to match training
                threshold=threshold,
                strict=False,  # Allow partial loading for architecture mismatches
            )
        except RuntimeError as e:
            # If loading fails due to key mismatches, try manual loading with filtering
            if "size mismatch" in str(e) or "Unexpected key" in str(e):
                print(f"Warning: Checkpoint has incompatible keys. Attempting partial load with filtering...")
                
                # Create model first with class weights
                model = MultiModalLightningModule(
                    config=config,
                    lr=lr,
                    warmup=None if warmup == -1 else warmup,
                    dinov3_checkpoint=None,
                    resnet_checkpoint=None,
                    freeze_dinov3=False,
                    freeze_resnet=False,
                    dinov3_model_name=dinov3_model_name,
                    class_weights=class_weights,  # Pass class weights to match training
                    threshold=threshold,
                )
                
                # Extract state dict from already-loaded checkpoint
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Filter out incompatible keys (but keep loss.pos_weight handling separate)
                filtered_state_dict = {}
                skipped_keys = []
                
                # Get model's state dict to check expected shapes
                model_state_dict = model.state_dict()
                
                for key, value in state_dict.items():
                    # Skip loss.pos_weight - it's already handled via class_weights parameter
                    # The loss function is created with class_weights in __init__, so we don't need to load it
                    if 'loss.pos_weight' in key:
                        # Already handled via class_weights parameter, skip from state_dict
                        continue
                    
                    # Skip classifier layers if dimensions don't match
                    if 'classifier' in key.lower():
                        if key in model_state_dict:
                            expected_shape = model_state_dict[key].shape
                            if value.shape != expected_shape:
                                skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {expected_shape})")
                                continue
                        else:
                            skipped_keys.append(f"{key} (not in model)")
                            continue
                    
                    # Include all other keys
                    filtered_state_dict[key] = value
                
                if skipped_keys:
                    print(f"Filtered out {len(skipped_keys)} incompatible keys:")
                    for key in skipped_keys[:10]:  # Show first 10
                        print(f"  - {key}")
                    if len(skipped_keys) > 10:
                        print(f"  ... and {len(skipped_keys) - 10} more")
                
                # Load filtered state dict
                missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
                
                if missing_keys:
                    print(f"Warning: {len(missing_keys)} keys were missing (using random initialization):")
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    if len(missing_keys) > 5:
                        print(f"  ... and {len(missing_keys) - 5} more")
                
                if unexpected_keys:
                    print(f"Info: {len(unexpected_keys)} unexpected keys were ignored")
            else:
                # Re-raise if it's a different error
                raise
        
        model.eval()

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Architecture: {architecture or 'unknown'}")
    print(f"Band config: {bandconfig or 'unknown'}")
    print(f"Channels: {hparams.get('channels', 'unknown')}")
    print(f"Threshold: {threshold}")
    print(f"{'=' * 60}\n")

    all_predictions, all_probabilities, all_labels = [], [], []
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    device = next(model.parameters()).device

    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            
            # Handle different model types
            # Multimodal models need split input: RGB and non-RGB/S1
            if hasattr(model, '_split_modalities'):
                rgb_data, non_rgb_s1_data = model._split_modalities(x)
                logits = model(rgb_data, non_rgb_s1_data)
            else:
                # BigEarthNet models can take the full tensor
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
    sample_ids = np.arange(len(predictions)) if return_sample_ids else None

    print(f"Collected predictions for {len(predictions)} samples")
    return predictions, probabilities, labels, sample_ids


__all__ = [
    "_infer_dinov3_model_from_checkpoint",
    "_detect_model_type",
    "load_model_and_infer",
]
