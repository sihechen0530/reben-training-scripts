"""
This script loads a multimodal checkpoint and evaluates it on the BigEarthNet v2.0 test set.
"""
import sys
from pathlib import Path

# Add parent directory to path to allow importing multimodal module
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import lightning.pytorch as pl
import torch
import typer
from configilm.extra.BENv2_utils import resolve_data_dir, STANDARD_BANDS
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet

from multimodal.lightning_module import MultiModalLightningModule
from scripts.utils import get_benv2_dir_dict, default_dm

__author__ = "BIFOLD/RSiM TU Berlin"


def _infer_dinov3_model_from_checkpoint(checkpoint_path: str) -> str:
    """
    Infer DINOv3 model name from checkpoint path or checkpoint contents.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Inferred DINOv3 model name
    """
    from pathlib import Path
    import torch
    
    # Strategy 1: Try to infer from filename
    ckpt_path = Path(checkpoint_path)
    filename = ckpt_path.name.lower()
    
    # Check filename for model size indicators
    if "dinov3-large" in filename or "dinov3-l" in filename or "dinov3l" in filename:
        return "facebook/dinov3-vitl16-pretrain-lvd1689m"
    elif "dinov3-base" in filename or "dinov3-b" in filename or "dinov3b" in filename:
        return "facebook/dinov3-vitb16-pretrain-lvd1689m"
    elif "dinov3-small" in filename or "dinov3-s" in filename or "dinov3s" in filename:
        return "facebook/dinov3-vits16-pretrain-lvd1689m"
    elif "dinov3-giant" in filename or "dinov3-g" in filename or "dinov3g" in filename:
        return "facebook/dinov3-vitg16-pretrain-lvd1689m"
    
    # Strategy 2: Try to load checkpoint and check hyperparameters
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Check hyperparameters
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            if 'dinov3_model_name' in hparams:
                return hparams['dinov3_model_name']
            if 'architecture' in hparams:
                arch = hparams['architecture'].lower()
                if 'large' in arch or 'l' in arch:
                    return "facebook/dinov3-vitl16-pretrain-lvd1689m"
                elif 'base' in arch or 'b' in arch:
                    return "facebook/dinov3-vitb16-pretrain-lvd1689m"
                elif 'small' in arch or 's' in arch:
                    return "facebook/dinov3-vits16-pretrain-lvd1689m"
                elif 'giant' in arch or 'g' in arch:
                    return "facebook/dinov3-vitg16-pretrain-lvd1689m"
        
        # Strategy 3: Check state_dict to infer model size from embedding dimension
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Look for embedding dimension in various keys
        hidden_size = None
        for key in list(state_dict.keys()):
            value = state_dict[key]
            if not hasattr(value, 'shape'):
                continue
                
            shape = value.shape
            if len(shape) < 2:
                continue
            
            # Check for cls_token, mask_token, or patch_embeddings
            if 'cls_token' in key.lower() or 'mask_token' in key.lower():
                if len(shape) >= 2:
                    hidden_size = shape[-1]
                    break
            elif 'patch_embeddings' in key.lower() and 'weight' in key.lower():
                if len(shape) == 4 and shape[1] == 3:
                    hidden_size = shape[0]
                    break
            elif 'embeddings' in key.lower() and 'weight' in key.lower():
                if len(shape) == 2:
                    hidden_size = shape[1]
                    break
        
        # Map hidden size to model name
        if hidden_size is not None:
            if hidden_size == 1024:
                return "facebook/dinov3-vitl16-pretrain-lvd1689m"
            elif hidden_size == 768:
                return "facebook/dinov3-vitb16-pretrain-lvd1689m"
            elif hidden_size == 384:
                return "facebook/dinov3-vits16-pretrain-lvd1689m"
            elif hidden_size == 1536:
                return "facebook/dinov3-vitg16-pretrain-lvd1689m"
    except Exception as e:
        print(f"Warning: Could not infer DINOv3 model from checkpoint: {e}")
    
    # Default to base if cannot infer
    return "facebook/dinov3-vitb16-pretrain-lvd1689m"


def main(
        checkpoint_path: str = typer.Option(..., help="Path to checkpoint file to test"),
        architecture: str = typer.Option(None, help="DINOv3 model size (dinov3-small/base/large/giant). "
                                                   "If None, will try to extract from checkpoint or infer from filename"),
        seed: int = typer.Option(42, help="Random seed (should match training seed)"),
        lr: float = typer.Option(0.001, help="Learning rate (not used for testing, but needed for model initialization)"),
        bs: int = typer.Option(32, help="Batch size"),
        drop_rate: float = typer.Option(0.15, help="Dropout rate (should match training)"),
        warmup: int = typer.Option(1000, help="Warmup steps (not used for testing, but needed for model initialization)"),
        workers: int = typer.Option(8, help="Number of workers"),
        bandconfig: str = typer.Option(None,
                                      help="Band configuration: rgb, s2, or s2s1/multimodal. "
                                           "If None, will try to extract from checkpoint"),
        use_s1: bool = typer.Option(None, "--use-s1/--no-use-s1", 
                                    help="Whether to include S1 (Sentinel-1) data. "
                                         "If None, will try to extract from checkpoint"),
        test_run: bool = typer.Option(False, help="Run testing with fewer batches (for quick testing)"),
        dinov3_model_name: str = typer.Option(None, help="DINOv3 HuggingFace model name (e.g., facebook/dinov3-base). "
                                                         "If None, will be inferred from architecture parameter or checkpoint."),
        threshold: float = typer.Option(0.5, help="Threshold for binary predictions in multi-label classification (default: 0.5)"),
):
    """
    Test a trained multimodal checkpoint on the BigEarthNet v2.0 test set.
    
    The checkpoint path should be a path to a .ckpt file saved by PyTorch Lightning.
    Example usage:
        python test_multimodal.py --checkpoint-path ./checkpoints/multimodal-model.ckpt
        python test_multimodal.py --checkpoint-path ./checkpoints/model.ckpt --architecture dinov3-base --bandconfig s2s1
    """
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    
    # Check if checkpoint exists
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to load hyperparameters from checkpoint if available
    checkpoint_channels = None
    checkpoint_use_s1 = None
    checkpoint_bandconfig = None
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'hyper_parameters' in checkpoint:
            hparams_ckpt = checkpoint['hyper_parameters']
            if architecture is None and 'architecture' in hparams_ckpt:
                architecture = hparams_ckpt['architecture']
                print(f"Loaded architecture from checkpoint: {architecture}")
            if bandconfig is None and 'bandconfig' in hparams_ckpt:
                checkpoint_bandconfig = hparams_ckpt['bandconfig']
                bandconfig = checkpoint_bandconfig
                print(f"Loaded bandconfig from checkpoint: {bandconfig}")
            if use_s1 is None and 'use_s1' in hparams_ckpt:
                checkpoint_use_s1 = hparams_ckpt['use_s1']
                use_s1 = checkpoint_use_s1
                print(f"Loaded use_s1 from checkpoint: {use_s1}")
            if 'channels' in hparams_ckpt:
                checkpoint_channels = hparams_ckpt['channels']
                print(f"Loaded channels from checkpoint: {checkpoint_channels}")
            if 'dropout' in hparams_ckpt and drop_rate == 0.15:
                drop_rate = hparams_ckpt['dropout']
                print(f"Loaded drop_rate from checkpoint: {drop_rate}")
            if 'seed' in hparams_ckpt and seed == 42:
                seed = hparams_ckpt['seed']
                print(f"Loaded seed from checkpoint: {seed}")
    except Exception as e:
        print(f"Warning: Could not load hyperparameters from checkpoint: {e}")
        print("Will try to infer from filename or use provided/default values.")
    
    # Determine DINOv3 model name
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
                dinov3_model_name = _infer_dinov3_model_from_checkpoint(checkpoint_path)
        else:
            dinov3_model_name = _infer_dinov3_model_from_checkpoint(checkpoint_path)
    
    print(f"Using DINOv3 model: {dinov3_model_name}")
    
    # Determine band configuration
    if bandconfig is None:
        # Try to infer from checkpoint channels
        if checkpoint_channels is not None:
            if checkpoint_channels == 3:
                bandconfig = "rgb"
            elif checkpoint_channels == 12:
                bandconfig = "s2"
            elif checkpoint_channels == 14:
                bandconfig = "s2s1"
            else:
                # Default based on channels
                if checkpoint_channels < 12:
                    bandconfig = "s2"
                else:
                    bandconfig = "s2s1"
            print(f"Inferred bandconfig from channels ({checkpoint_channels}): {bandconfig}")
        else:
            # Default to s2s1 if cannot infer
            bandconfig = "s2s1"
            print(f"Warning: Could not infer bandconfig. Using default: {bandconfig}")
    
    # Normalize bandconfig
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
    if band_alias not in band_map:
        raise ValueError(
            "Unsupported bandconfig. Choose from rgb, s2, s2s1/multimodal."
        )
    effective_bandconfig = band_map[band_alias]
    
    # Determine use_s1
    if use_s1 is None:
        use_s1 = (effective_bandconfig == "s2s1")
        if checkpoint_use_s1 is not None:
            use_s1 = checkpoint_use_s1
            print(f"Using use_s1 from checkpoint: {use_s1}")
        else:
            print(f"Inferred use_s1 from bandconfig: {use_s1}")
    
    # FIXED MODEL PARAMETERS
    num_classes = 19
    img_size = 120
    
    # Set seed
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    # ============================================================================
    # CONFIGURE BAND ORDER AND CALCULATE RESNET INPUT CHANNELS
    # ============================================================================
    # Get S2 and S1 bands from STANDARD_BANDS
    s2_bands = STANDARD_BANDS.get("S2", STANDARD_BANDS.get("s2_full", []))
    s1_bands = STANDARD_BANDS.get("S1", ["VV", "VH"])
    
    # Define RGB bands (must be first 3 channels for DINOv3)
    rgb_bands = ["B04", "B03", "B02"]
    full_s2_non_rgb = [b for b in s2_bands if b not in rgb_bands]
    
    if effective_bandconfig == "rgb":
        s2_non_rgb = []
    else:
        s2_non_rgb = full_s2_non_rgb
    
    # Reorder S2 bands: RGB first, then non-RGB (if any)
    s2_ordered = rgb_bands + s2_non_rgb
    
    # Calculate ResNet input channels: S2 non-RGB + optionally S1
    resnet_input_channels = len(s2_non_rgb) + (len(s1_bands) if use_s1 else 0)
    use_resnet = resnet_input_channels > 0
    
    # Combine: S2 (ordered: RGB + non-RGB) + optionally S1
    if effective_bandconfig == "rgb":
        multimodal_bands = rgb_bands
        config_description = "RGB only"
    elif use_s1:
        multimodal_bands = s2_ordered + s1_bands
        config_description = "Multimodal (S2 ordered + S1)"
    else:
        multimodal_bands = s2_ordered
        config_description = "Multimodal (S2 only, no S1)"
    
    num_channels = len(multimodal_bands)
    
    # Register with BENv2DataSet
    STANDARD_BANDS[num_channels] = multimodal_bands
    STANDARD_BANDS["multimodal"] = multimodal_bands
    BENv2DataSet.channel_configurations[num_channels] = multimodal_bands
    BENv2DataSet.avail_chan_configs[num_channels] = config_description
    
    # Validate channels match checkpoint if we extracted them
    if checkpoint_channels is not None and num_channels != checkpoint_channels:
        print(f"Warning: Channel mismatch! Checkpoint has {checkpoint_channels} channels, "
              f"but bandconfig '{effective_bandconfig}' produces {num_channels} channels.")
        print(f"Proceeding with {num_channels} channels...")
    
    print(f"\nUsing band configuration: {config_description} (source: {effective_bandconfig})")
    print(f"Number of channels: {num_channels}")
    print(f"S1 Usage: {'ENABLED' if use_s1 else 'DISABLED'}")
    print(f"ResNet branch: {'ENABLED' if use_resnet else 'DISABLED'} (input channels: {resnet_input_channels})")
    print(f"Band order: {multimodal_bands}")
    
    # ============================================================================
    # CREATE MODEL CONFIGURATION
    # ============================================================================
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
        "fusion": {
            "type": "concat",  # Default, can be overridden if in checkpoint
        },
        "classifier": {
            "type": "linear",  # Default, can be overridden if in checkpoint
            "num_classes": num_classes,
            "drop_rate": drop_rate,
        },
        "image_size": img_size,
        "rgb_band_names": rgb_bands,
        "s2_band_order": s2_ordered,
        "s1_band_order": s1_bands if use_s1 else [],
        "use_s1": use_s1,
    }
    
    # Try to load fusion and classifier config from checkpoint
    try:
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            if 'fusion_type' in hparams:
                config["fusion"]["type"] = hparams['fusion_type']
                print(f"Loaded fusion_type from checkpoint: {config['fusion']['type']}")
            if 'classifier_type' in hparams:
                config["classifier"]["type"] = hparams['classifier_type']
                print(f"Loaded classifier_type from checkpoint: {config['classifier']['type']}")
            if 'classifier_hidden_dim' in hparams and config["classifier"]["type"] == "mlp":
                config["classifier"]["hidden_dim"] = hparams['classifier_hidden_dim']
                print(f"Loaded classifier_hidden_dim from checkpoint: {config['classifier']['hidden_dim']}")
    except:
        pass
    
    # Create model instance (weights will be loaded from checkpoint)
    model = MultiModalLightningModule(
        config=config,
        lr=lr,
        warmup=None if warmup == -1 else warmup,
        dinov3_checkpoint=None,  # Will load from main checkpoint
        resnet_checkpoint=None,  # Will load from main checkpoint
        freeze_dinov3=False,
        freeze_resnet=False,
        dinov3_model_name=dinov3_model_name,
        threshold=threshold,
    )
    
    hparams = {
        "architecture": architecture or "multimodal",
        "seed": seed,
        "lr": lr,
        "epochs": 0,  # Not used for testing
        "batch_size": bs,
        "workers": workers,
        "channels": num_channels,
        "dropout": drop_rate,
        "bandconfig": effective_bandconfig,
        "warmup": warmup,
        "use_s1": use_s1,
        "dinov3_model_name": dinov3_model_name,
    }
    
    # Create trainer for testing (no callbacks needed for testing)
    trainer = pl.Trainer(
        limit_test_batches=5 if test_run else None,
        accelerator="auto",
        logger=False,  # Disable logging for testing
    )
    
    # Set up data
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
    dm = default_dm(hparams, data_dirs, img_size)
    
    print(f"\n{'='*60}")
    print(f"Testing checkpoint: {checkpoint_path}")
    print(f"Architecture: {architecture or 'multimodal'}")
    print(f"Band config: {config_description} ({num_channels} channels)")
    print(f"DINOv3 model: {dinov3_model_name}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}\n")
    
    # Test the checkpoint
    results = trainer.test(model, datamodule=dm, ckpt_path=str(ckpt_path))
    
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    for key, value in results[0].items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    typer.run(main)

