"""
This script loads a checkpoint and evaluates it on the BigEarthNet v2.0 test set.
"""
import sys
from pathlib import Path

# Add parent directory to path to allow importing reben_publication
# This allows running from the scripts directory
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import lightning.pytorch as pl
import torch
import typer
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BENv2_utils import resolve_data_dir

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from scripts.utils import get_benv2_dir_dict, get_bands, default_trainer, default_dm

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def main(
        checkpoint_path: str = typer.Option(..., help="Path to checkpoint file to test"),
        architecture: str = typer.Option(None, help="Model name (timm model name or dinov3-base/dinov3-large/dinov3-small/dinov3-giant). "
                                                   "If None, will try to extract from checkpoint or infer from filename"),
        seed: int = typer.Option(42, help="Random seed (should match training seed)"),
        lr: float = typer.Option(0.001, help="Learning rate (not used for testing, but needed for model initialization)"),
        bs: int = typer.Option(32, help="Batch size"),
        drop_rate: float = typer.Option(0.15, help="Dropout rate (should match training)"),
        drop_path_rate: float = typer.Option(0.15, help="Drop path rate (should match training)"),
        warmup: int = typer.Option(1000, help="Warmup steps (not used for testing, but needed for model initialization)"),
        workers: int = typer.Option(8, help="Number of workers"),
        bandconfig: str = typer.Option(None,
                                      help="Band configuration, one of all, s2, s1, rgb, all_full, s2_full, s1_full. "
                                           "rgb uses 3-channel RGB (B04, B03, B02) from Sentinel-2. "
                                           "If None, will try to extract from checkpoint filename"),
        test_run: bool = typer.Option(False, help="Run testing with fewer batches (for quick testing)"),
        dinov3_model_name: str = typer.Option(None, help="DINOv3 HuggingFace model name (e.g., facebook/dinov3-base). "
                                                         "If None, will be inferred from architecture parameter."),
        threshold: float = typer.Option(0.5, help="Threshold for binary predictions in multi-label classification (default: 0.5)"),
):
    """
    Test a trained checkpoint on the BigEarthNet v2.0 test set.
    
    The checkpoint path should be a path to a .ckpt file saved by PyTorch Lightning.
    Example usage:
        python test_checkpoint_BigEarthNetv2_0.py --checkpoint-path ./checkpoints/resnet18-42-12-val_mAP_macro-0.85.ckpt
        python test_checkpoint_BigEarthNetv2_0.py --checkpoint-path ./checkpoints/model.ckpt --threshold 0.3
    """
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    
    # Check if checkpoint exists
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to load hyperparameters from checkpoint if available
    checkpoint_channels = None
    checkpoint_data = None
    try:
        checkpoint_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        checkpoint = checkpoint_data
        if 'hyper_parameters' in checkpoint:
            hparams_ckpt = checkpoint['hyper_parameters']
            if architecture is None and 'architecture' in hparams_ckpt:
                architecture = hparams_ckpt['architecture']
                print(f"Loaded architecture from checkpoint: {architecture}")
            if bandconfig is None and 'bandconfig' in hparams_ckpt:
                bandconfig = hparams_ckpt['bandconfig']
                print(f"Loaded bandconfig from checkpoint: {bandconfig}")
            # Try to get channels directly from hyperparameters
            if 'channels' in hparams_ckpt:
                checkpoint_channels = hparams_ckpt['channels']
                print(f"Loaded channels from checkpoint: {checkpoint_channels}")
            if 'dropout' in hparams_ckpt and drop_rate == 0.15:  # Only override if using default
                drop_rate = hparams_ckpt['dropout']
                print(f"Loaded drop_rate from checkpoint: {drop_rate}")
            if 'drop_path_rate' in hparams_ckpt and drop_path_rate == 0.15:  # Only override if using default
                drop_path_rate = hparams_ckpt['drop_path_rate']
                print(f"Loaded drop_path_rate from checkpoint: {drop_path_rate}")
            if 'seed' in hparams_ckpt and seed == 42:  # Only override if using default
                seed = hparams_ckpt['seed']
                print(f"Loaded seed from checkpoint: {seed}")
            
            # Check if checkpoint was trained with class weights
            if 'use_balanced_weights' in hparams_ckpt and hparams_ckpt['use_balanced_weights']:
                print("Note: Checkpoint was trained with balanced class weights")
                # We don't need to recreate the weights for testing, just note it
        
        # If we still don't have channels or bandconfig, try to extract from state_dict
        if checkpoint_channels is None or bandconfig is None:
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Try to find the patch embeddings weight to determine input channels
                # This works for both DINOv3 and timm models
                # DINOv3 keys: model.backbone.embeddings.patch_embeddings.projection.weight or 
                #             model.model.backbone.embeddings.patch_embeddings.weight or
                #             backbone.embeddings.patch_embeddings.projection.weight
                # Timm keys: model.conv1.weight or model.stem.0.weight or model.model.conv1.weight
                
                # Priority order: check DINOv3 keys first, then timm keys
                search_patterns = [
                    'patch_embeddings.projection.weight',  # DINOv3 projection layer
                    'patch_embeddings.weight',            # DINOv3 patch embeddings
                    'conv1.weight',                       # Timm models
                    'stem.0.weight',                     # Timm EfficientNet-style
                ]
                
                for pattern in search_patterns:
                    for key in state_dict.keys():
                        if pattern in key:
                            weight_shape = state_dict[key].shape
                            if len(weight_shape) == 4:  # Conv weight: [out_channels, in_channels, H, W]
                                checkpoint_channels = weight_shape[1]
                                print(f"Extracted channels from checkpoint state_dict: {checkpoint_channels} (from {key})")
                                break
                    if checkpoint_channels is not None:
                        break
    except Exception as e:
        print(f"Warning: Could not load hyperparameters from checkpoint: {e}")
        print("Will try to infer from filename or use provided/default values.")
    
    # Try to extract architecture and bandconfig from checkpoint filename if still not provided
    # Checkpoint filename format: {architecture}-{seed}-{channels}-val_mAP_macro-{val_mAP:.2f}.ckpt
    if architecture is None or bandconfig is None:
        filename_parts = ckpt_path.stem.split('-')
        if len(filename_parts) >= 3 and architecture is None:
            # Try to extract architecture (first part)
            potential_arch = filename_parts[0]
            # Check if it's a valid architecture name (simple heuristic)
            if potential_arch and not potential_arch.isdigit():
                architecture = potential_arch
                print(f"Inferred architecture from filename: {architecture}")
        
        # Try to infer bandconfig from channels (3rd part) or from checkpoint_channels
        if bandconfig is None:
            channels = None
            # First try to use checkpoint_channels if we extracted it
            if checkpoint_channels is not None:
                channels = checkpoint_channels
                print(f"Using channels extracted from checkpoint: {channels}")
            else:
                # Fall back to filename parsing
                try:
                    channels = int(filename_parts[2])
                    print(f"Inferred channels from filename: {channels}")
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse channels from filename.")
                    channels = None
            
            if channels is not None:
                # Map channels to bandconfig (common mappings)
                # Note: For 3 channels, we default to 'rgb' but it could also be 's1_full' - 
                # the user should specify explicitly if ambiguous
                if channels == 12:
                    bandconfig = "all"
                elif channels == 10:
                    bandconfig = "s2"
                elif channels == 2:
                    bandconfig = "s1"
                elif channels == 13:
                    bandconfig = "all_full"
                elif channels == 11:
                    bandconfig = "s2_full"
                elif channels == 3:
                    # Default to rgb for 3 channels (common for RGB training)
                    # User can override if it's actually s1_full
                    bandconfig = "rgb"
                    print(f"Note: Inferred 'rgb' for 3 channels. If this is actually 's1_full', please specify --bandconfig explicitly.")
                else:
                    print(f"Warning: Unknown channel count {channels}. Attempting to use 's2' for {channels} channels.")
                    # Default to s2 for 10 channels, but warn if it's something else
                    if channels == 10:
                        bandconfig = "s2"
                    else:
                        print(f"Error: Cannot determine bandconfig for {channels} channels. Please specify --bandconfig explicitly.")
                        bandconfig = None
                if bandconfig:
                    print(f"Inferred bandconfig from channels ({channels}): {bandconfig}")
    
    if architecture is None:
        raise ValueError("Could not infer architecture from checkpoint filename. Please provide --architecture explicitly.")
    if bandconfig is None:
        raise ValueError("Could not infer bandconfig from checkpoint filename. Please provide --bandconfig explicitly.")
    
    # FIXED MODEL PARAMETERS
    num_classes = 19
    img_size = 120
    
    # set seed
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    bands, channels = get_bands(bandconfig)
    
    # Validate channels match checkpoint if we extracted them
    if checkpoint_channels is not None and channels != checkpoint_channels:
        raise ValueError(
            f"Channel mismatch! Checkpoint has {checkpoint_channels} channels, but bandconfig '{bandconfig}' "
            f"produces {channels} channels. Please specify the correct --bandconfig that matches your training configuration."
        )
    
    print(f"\nUsing band configuration: {bandconfig}")
    print(f"Number of channels: {channels}")
    print(f"Bands: {bands}")
    
    # fixed model parameters based on the BigEarthNet v2.0 dataset
    config = ILMConfiguration(
        network_type=ILMType.IMAGE_CLASSIFICATION,
        classes=num_classes,
        image_size=img_size,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        timm_model_name=architecture,
        channels=channels,
    )
    warmup = None if warmup == -1 else warmup
    
    # Determine DINOv3 model name if needed
    dinov3_name = dinov3_model_name
    if architecture.startswith('dinov3') and dinov3_name is None:
        # Map architecture names to HuggingFace model names
        if 'small' in architecture.lower() or 's' in architecture.lower():
            dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        elif 'base' in architecture.lower() or 'b' in architecture.lower():
            dinov3_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        elif 'large' in architecture.lower() or 'l' in architecture.lower():
            dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        elif 'giant' in architecture.lower() or 'g' in architecture.lower():
            dinov3_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
        else:
            dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"  # default to small
    
    # Create model instance (weights will be loaded from checkpoint)
    model = BigEarthNetv2_0_ImageClassifier(config, lr=lr, warmup=warmup, dinov3_model_name=dinov3_name, threshold=threshold)
    
    # Manually load checkpoint state_dict, excluding loss module to avoid pos_weight mismatch
    # This is necessary because checkpoints trained with class weights have loss.pos_weight
    # but we create the model without class weights for testing
    try:
        # Reuse checkpoint_data if already loaded, otherwise load it
        if checkpoint_data is None:
            checkpoint_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        checkpoint = checkpoint_data
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Filter out loss module keys to avoid pos_weight mismatch
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('loss.')}
            if len(filtered_state_dict) < len(state_dict):
                print(f"Note: Excluding {len(state_dict) - len(filtered_state_dict)} loss module keys from checkpoint (pos_weight mismatch)")
            # Load the filtered state dict
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                print(f"Warning: {len(missing_keys)} keys not found in model (this may be normal)")
            if unexpected_keys:
                print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint (this may be normal)")
        else:
            print("Warning: Checkpoint does not contain 'state_dict', attempting to load directly")
            # Try loading as direct state dict
            filtered_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('loss.')}
            model.load_state_dict(filtered_state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Could not manually load checkpoint state_dict: {e}")
        print("Will attempt to load via PyTorch Lightning (may fail if loss.pos_weight mismatch)")
    
    hparams = {
        "architecture": architecture,
        "seed": seed,
        "lr": lr,
        "epochs": 0,  # Not used for testing
        "batch_size": bs,
        "workers": workers,
        "channels": channels,
        "dropout": drop_rate,
        "drop_path_rate": drop_path_rate,
        "bandconfig": bandconfig,
        "warmup": warmup,
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
    print(f"Architecture: {architecture}")
    print(f"Band config: {bandconfig} ({channels} channels)")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}\n")
    
    # Test the model (weights already loaded manually, don't pass ckpt_path)
    results = trainer.test(model, datamodule=dm)
    
    print(f"\n{'='*60}")
    print("Test Results:")
    print(f"{'='*60}")
    for key, value in results[0].items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    typer.run(main)

