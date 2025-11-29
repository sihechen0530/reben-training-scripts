"""
Utility helpers for loading BigEarthNet models for diagnosis scripts.

Supports three model types:
1. HuggingFace pretrained models (model_name string)
2. Local BigEarthNet checkpoints (.ckpt file)
3. Local Multimodal checkpoints (.ckpt file)
"""

import sys
import json
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
    split: str = "test",
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
        try:
            model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_path)
        except AttributeError as e:
            if "items" in str(e) and ("ILMConfiguration" in str(e) or "config" in str(e).lower()):
                # Workaround: Manually load the model to bypass the config.items() issue
                print("Attempting workaround for config serialization issue...")
                try:
                    from huggingface_hub import snapshot_download
                    import tempfile
                    import json
                    
                    # Download model files
                    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                    local_dir = snapshot_download(repo_id=model_path, cache_dir=cache_dir)
                    local_path = Path(local_dir)
                    
                    # Load config.json and convert to ILMConfiguration
                    config_file = local_path / "config.json"
                    if not config_file.exists():
                        raise FileNotFoundError(f"Config file not found in {local_path}")
                    
                    with open(config_file, 'r') as f:
                        config_dict = json.load(f)
                    
                    # Convert dict to ILMConfiguration
                    # Handle both flat and nested config structures
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
                    
                    # Load model weights
                    model_file = local_path / "pytorch_model.bin"
                    if not model_file.exists():
                        model_file = local_path / "model.safetensors"
                    
                    if not model_file.exists():
                        raise FileNotFoundError(f"Model weights not found in {local_path}")
                    
                    # Create model with config
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
                    
                    # Load weights
                    if model_file.suffix == '.safetensors':
                        from safetensors.torch import load_file
                        state_dict = load_file(str(model_file))
                    else:
                        state_dict = torch.load(model_file, map_location='cpu')
                    
                    # Handle nested state_dict (might have 'state_dict' key)
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    
                    # Load state dict, handling potential key mismatches
                    model.load_state_dict(state_dict, strict=False)
                    print("Successfully loaded model using workaround method")
                    
                except Exception as e2:
                    raise ValueError(
                        f"Failed to load HuggingFace model '{model_path}' due to config serialization issue. "
                        f"Workaround also failed.\n\n"
                        f"Original error: {e}\n"
                        f"Workaround error: {e2}\n\n"
                        f"Possible solutions:\n"
                        f"1. Check if the model was saved with a compatible version of the code\n"
                        f"2. Try updating huggingface_hub: pip install --upgrade huggingface_hub\n"
                        f"3. Contact the model maintainer about the config format issue"
                    ) from e2
            else:
                raise
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

        model = BigEarthNetv2_0_ImageClassifier.load_from_checkpoint(
            str(ckpt_path),
            config=config,
            lr=lr,
            warmup=warmup,
            dinov3_model_name=dinov3_name,
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
            threshold=threshold,
        )
        model.eval()

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Architecture: {architecture or 'unknown'}")
    print(f"Band config: {bandconfig or 'unknown'}")
    print(f"Channels: {hparams.get('channels', 'unknown')}")
    print(f"Threshold: {threshold}")
    print(f"{'=' * 60}\n")

    all_predictions, all_probabilities, all_labels = [], [], []
    if split == "validation":
        # Use "fit" stage to set up both train and validation datasets
        dm.setup(stage="fit")
        data_loader = dm.val_dataloader()
        split_name = "validation set"
        print(f"DEBUG: val_dataloader() returned: {type(data_loader)}, value: {data_loader}")
        if data_loader is None:
            raise ValueError(
                "Validation dataloader is None. The datamodule may not have validation data configured. "
                "Please ensure your dataset has a validation split."
            )
    else:
        dm.setup(stage="test")
        data_loader = dm.test_dataloader()
        split_name = "test set"
        if data_loader is None:
            raise ValueError(
                "Test dataloader is None. The datamodule may not have test data configured."
            )
    device = next(model.parameters()).device

    # Additional check: ensure dataloader is iterable
    try:
        # Try to get length or check if it's iterable
        if hasattr(data_loader, '__len__'):
            try:
                dl_len = len(data_loader)
                print(f"DataLoader has {dl_len} batches")
            except (TypeError, AttributeError):
                print("DataLoader length is not available (might be a generator)")
    except Exception as e:
        raise ValueError(
            f"Error checking dataloader: {e}. "
            f"The {split_name} dataloader may not be properly configured."
        ) from e

    print(f"Running inference on {split_name}...")
    print(f"Device: {device}, Batch size: {bs}, Workers: {workers}")
    
    # Performance optimization: Use non_blocking transfers and keep tensors on GPU longer
    try:
        with torch.no_grad():
            # Enable optimizations for inference
            if torch.cuda.is_available() and device.type == 'cuda':
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            
            for batch_idx, (x, y) in enumerate(data_loader):
                # Use non_blocking transfer for faster CPU->GPU
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                logits = model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).long()

                # Keep on GPU and move to CPU asynchronously (batch the CPU transfers)
                # This reduces GPU-CPU synchronization overhead
                all_predictions.append(preds.detach().cpu())
                all_probabilities.append(probs.detach().cpu())
                all_labels.append(y.detach().cpu())

                if test_run and batch_idx >= 4:
                    break
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")
    except TypeError as e:
        if "NoneType" in str(e) and "len" in str(e):
            raise ValueError(
                f"Failed to iterate over {split_name} dataloader. "
                f"The dataloader may be None or not properly initialized. "
                f"Original error: {e}\n"
                f"Please ensure your dataset has a {split} split configured."
            ) from e
        raise

    if len(all_predictions) == 0:
        raise ValueError(
            f"No data was collected from {split_name}. "
            f"The dataloader may be empty or no batches were processed."
        )

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
