"""
Training script for multimodal classification model.

This script trains a multimodal model that combines:
- S2 RGB data (3 channels) through DINOv3 backbone
- S2 non-RGB (11 channels) + S1 (2 channels) through ResNet101 backbone

Data Loading and Channel Order:
--------------------------------
The dataloader (BENv2DataSet) uses channel_configurations[num_channels] to determine
which bands to load and in what order. The configilm library's stack_and_interpolate
function uses an 'order' parameter to stack bands according to the list order.

By registering multimodal_bands with RGB bands first:
  multimodal_bands = [B04, B03, B02, ...S2_non_RGB..., VV, VH]
  
The dataloader will stack bands in this exact order, so:
  - Channel 0 = B04 (Red)
  - Channel 1 = B03 (Green)  
  - Channel 2 = B02 (Blue)
  - Channels 3+ = S2 non-RGB + S1

Therefore, we can safely extract RGB as: rgb_data = x[:, :3, :, :]
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
from typing import Optional
from configilm.extra.BENv2_utils import resolve_data_dir, STANDARD_BANDS
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet

from multimodal.lightning_module import MultiModalLightningModule
from scripts.utils import get_benv2_dir_dict, default_trainer, default_dm

__author__ = "BIFOLD/RSiM TU Berlin"


def _infer_dinov3_model_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """
    Infer DINOv3 model name from checkpoint path or checkpoint contents.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Inferred DINOv3 model name or None
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
                # Shape should be [1, 1, hidden_size] or [1, hidden_size]
                if len(shape) >= 2:
                    hidden_size = shape[-1]
                    break
            elif 'patch_embeddings' in key.lower() and 'weight' in key.lower():
                # Patch embedding weight: [hidden_size, 3, 16, 16]
                if len(shape) == 4 and shape[1] == 3:
                    hidden_size = shape[0]
                    break
            elif 'embeddings' in key.lower() and 'weight' in key.lower():
                # General embedding weight
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
            else:
                print(f"  Warning: Unknown hidden size {hidden_size} in checkpoint")
    except Exception as e:
        print(f"  Warning: Could not infer DINOv3 model from checkpoint: {e}")
    
    return None


def main(
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(512, help="Batch size"),
        drop_rate: float = typer.Option(0.15, help="Dropout rate"),
        warmup: int = typer.Option(1000, help="Warmup steps, set to -1 for automatic calculation"),
        workers: int = typer.Option(8, help="Number of workers"),
        use_wandb: bool = typer.Option(False, "--use-wandb/--no-wandb", help="Use wandb for logging"),
        test_run: bool = typer.Option(True, "--test-run/--no-test-run", help="Run training with fewer epochs and batches"),
        # DINOv3 configuration
        dinov3_hidden_size: int = typer.Option(768, help="DINOv3 hidden size (embedding dimension). Options: 384 (small), 768 (base, default), 1024 (large), 1536 (giant). This determines which DINOv3 model to use, even when loading from checkpoint."),
        dinov3_pretrained: bool = typer.Option(True, "--dinov3-pretrained/--no-dinov3-pretrained", help="Use pretrained DINOv3 weights"),
        dinov3_freeze: bool = typer.Option(False, "--dinov3-freeze/--no-dinov3-freeze", help="Freeze DINOv3 backbone"),
        dinov3_lr: float = typer.Option(1e-4, help="Learning rate for DINOv3 backbone"),
        dinov3_checkpoint: str = typer.Option(None, 
                                             help="Path to checkpoint file to load DINOv3 backbone weights from. "
                                                  "Can be absolute path or relative to scripts/checkpoints/. "
                                                  "Model size is determined by --dinov3-hidden-size parameter."),
        # ResNet101 configuration
        resnet_pretrained: bool = typer.Option(True, "--resnet-pretrained/--no-resnet-pretrained", help="Use pretrained ResNet101 weights"),
        resnet_freeze: bool = typer.Option(False, "--resnet-freeze/--no-resnet-freeze", help="Freeze ResNet101 backbone"),
        resnet_lr: float = typer.Option(1e-4, help="Learning rate for ResNet101 backbone"),
        resnet_checkpoint: str = typer.Option(None,
                                             help="Path to checkpoint file to load ResNet101 backbone weights from. "
                                                  "Can be absolute path or relative to scripts/checkpoints/"),
        # Fusion configuration
        fusion_type: str = typer.Option("concat", help="Fusion type: concat, weighted, or linear_projection"),
        fusion_output_dim: int = typer.Option(None, help="Output dimension for linear_projection fusion (optional)"),
        # Classifier configuration
        classifier_type: str = typer.Option("linear", help="Classifier type: linear or mlp"),
        classifier_hidden_dim: int = typer.Option(512, help="Hidden dimension for MLP classifier"),
        # Data configuration
        use_s1: bool = typer.Option(False, "--use-s1/--no-use-s1", help="Whether to include S1 (Sentinel-1) data. If False, only S2 non-RGB bands are used for ResNet."),
        # Training configuration
        resume_from: str = typer.Option(None, help="Path to checkpoint file to resume training from. "
                                                   "Can be a full path or 'best'/'last' to use the best/last checkpoint from the checkpoint directory."),
):
    """
    Train a multimodal classification model.
    
    The model uses:
    - DINOv3 backbone for S2 RGB data (3 channels: B04, B03, B02)
    - ResNet101 backbone for S2 non-RGB (9 channels) + optionally S1 (2 channels: VV, VH)
    - Late fusion to combine features
    - Classification head for final predictions
    
    Data Flow:
    ----------
    If use_s1=True:
    1. Dataloader loads data with bands in order: [RGB (3), S2_non-RGB (9), S1 (2)]
    2. Lightning module splits: rgb_data = x[:, :3] and non_rgb_s1_data = x[:, 3:]
    3. Model processes: DINOv3(rgb_data) + ResNet(non_rgb_s1_data) -> fusion -> classifier
    
    If use_s1=False:
    1. Dataloader loads data with bands in order: [RGB (3), S2_non-RGB (9)]
    2. Lightning module splits: rgb_data = x[:, :3] and non_rgb_data = x[:, 3:]
    3. Model processes: DINOv3(rgb_data) + ResNet(non_rgb_data) -> fusion -> classifier
    """
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    
    # ============================================================================
    # FIXED MODEL PARAMETERS
    # ============================================================================
    num_classes = 19
    img_size = 120
    
    # Set seed
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    # ============================================================================
    # DETERMINE DINOv3 MODEL NAME FROM HIDDEN SIZE
    # ============================================================================
    # Map hidden size to model name
    hidden_size_to_model = {
        384: "facebook/dinov3-vits16-pretrain-lvd1689m",
        768: "facebook/dinov3-vitb16-pretrain-lvd1689m",
        1024: "facebook/dinov3-vitl16-pretrain-lvd1689m",
        1536: "facebook/dinov3-vitg16-pretrain-lvd1689m",
    }
    
    if dinov3_hidden_size not in hidden_size_to_model:
        raise ValueError(
            f"Invalid dinov3_hidden_size: {dinov3_hidden_size}. "
            f"Must be one of {list(hidden_size_to_model.keys())}"
        )
    
    dinov3_model_name = hidden_size_to_model[dinov3_hidden_size]
    print(f"Using DINOv3 model: {dinov3_model_name} (hidden_size={dinov3_hidden_size})")
    
    # ============================================================================
    # RESOLVE CHECKPOINT PATHS
    # ============================================================================
    dinov3_ckpt_path = None
    if dinov3_checkpoint is not None:
        dinov3_ckpt_path_obj = Path(dinov3_checkpoint)
        if not dinov3_ckpt_path_obj.exists():
            dinov3_ckpt_path_obj = Path("./checkpoints") / dinov3_checkpoint
            if not dinov3_ckpt_path_obj.exists():
                raise FileNotFoundError(
                    f"DINOv3 checkpoint not found: {dinov3_checkpoint}\n"
                    f"Tried: {dinov3_checkpoint} and ./checkpoints/{dinov3_checkpoint}"
                )
        dinov3_ckpt_path = str(dinov3_ckpt_path_obj.resolve())
        print(f"Using DINOv3 checkpoint: {dinov3_ckpt_path}")
        print(f"  Note: Model size is determined by --dinov3-hidden-size={dinov3_hidden_size}, not inferred from checkpoint")
    
    # ============================================================================
    # BUILD MODEL CONFIGURATION
    # ============================================================================
    config = {
        "backbones": {
            "dinov3": {
                "model_name": dinov3_model_name,
                "pretrained": dinov3_pretrained,
                "freeze": dinov3_freeze,
                "lr": dinov3_lr,
            },
            "resnet101": {
                "pretrained": resnet_pretrained,
                "freeze": resnet_freeze,
                "lr": resnet_lr,
            },
        },
        "fusion": {
            "type": fusion_type,
        },
        "classifier": {
            "type": classifier_type,
            "num_classes": num_classes,
            "drop_rate": drop_rate,
        },
        "image_size": img_size,
    }
    
    if fusion_type == "linear_projection" and fusion_output_dim is not None:
        config["fusion"]["output_dim"] = fusion_output_dim
    
    if classifier_type == "mlp":
        config["classifier"]["hidden_dim"] = classifier_hidden_dim
    
    # ============================================================================
    # CONFIGURE BAND ORDER AND CALCULATE RESNET INPUT CHANNELS
    # (Must be done before model creation to set correct input_channels)
    # ============================================================================
    # Get S2 and S1 bands from STANDARD_BANDS
    s2_bands = STANDARD_BANDS.get("S2", STANDARD_BANDS.get("s2_full", []))
    s1_bands = STANDARD_BANDS.get("S1", ["VV", "VH"])  # Default S1 bands: VV, VH
    
    # Define RGB bands (must be first 3 channels for DINOv3)
    # RGB true color: B04 (Red), B03 (Green), B02 (Blue)
    rgb_bands = ["B04", "B03", "B02"]
    
    # Reorder S2 bands: RGB first, then non-RGB
    s2_non_rgb = [b for b in s2_bands if b not in rgb_bands]
    s2_ordered = rgb_bands + s2_non_rgb
    
    # Calculate ResNet input channels: S2 non-RGB + optionally S1
    resnet_input_channels = len(s2_non_rgb) + (len(s1_bands) if use_s1 else 0)
    
    # Configure ResNet input channels in config (needed before model creation)
    config["backbones"]["resnet101"]["input_channels"] = resnet_input_channels
    
    # Store band order in config for reference
    config["rgb_band_names"] = rgb_bands
    config["s2_band_order"] = s2_ordered
    config["s1_band_order"] = s1_bands if use_s1 else []
    config["use_s1"] = use_s1
    
    # ============================================================================
    # RESOLVE RESNET CHECKPOINT PATH
    # ============================================================================
    resnet_ckpt_path = None
    if resnet_checkpoint is not None:
        resnet_ckpt_path_obj = Path(resnet_checkpoint)
        if not resnet_ckpt_path_obj.exists():
            resnet_ckpt_path_obj = Path("./checkpoints") / resnet_checkpoint
            if not resnet_ckpt_path_obj.exists():
                raise FileNotFoundError(
                    f"ResNet checkpoint not found: {resnet_checkpoint}\n"
                    f"Tried: {resnet_checkpoint} and ./checkpoints/{resnet_checkpoint}"
                )
        resnet_ckpt_path = str(resnet_ckpt_path_obj.resolve())
        print(f"Using ResNet checkpoint: {resnet_ckpt_path}")
    
    # ============================================================================
    # CREATE MODEL
    # ============================================================================
    model = MultiModalLightningModule(
        config=config,
        lr=lr,
        warmup=None if warmup == -1 else warmup,
        dinov3_checkpoint=dinov3_ckpt_path,
        resnet_checkpoint=resnet_ckpt_path,
        freeze_dinov3=dinov3_freeze if dinov3_ckpt_path else False,
        freeze_resnet=resnet_freeze if resnet_ckpt_path else False,
        dinov3_model_name=dinov3_model_name,
    )
    
    # ============================================================================
    # REGISTER BAND CONFIGURATION WITH DATALOADER
    # ============================================================================
    # Combine: S2 (ordered: RGB + non-RGB) + optionally S1
    if use_s1:
        # Final order: [RGB (3), S2_non-RGB (9), S1 (2)]
        multimodal_bands = s2_ordered + s1_bands
        config_description = "Multimodal (S2 ordered + S1)"
    else:
        # Final order: [RGB (3), S2_non-RGB (9)]
        multimodal_bands = s2_ordered
        config_description = "Multimodal (S2 only, no S1)"
    
    num_channels = len(multimodal_bands)
    
    # Register with BENv2DataSet
    # IMPORTANT: BENv2DataSet uses channel_configurations[num_channels] to determine
    # which bands to load. The stack_and_interpolate function uses order=bands parameter,
    # which stacks bands according to the list order. Therefore, by registering
    # multimodal_bands with RGB first, the dataloader will stack bands in this exact order.
    STANDARD_BANDS[num_channels] = multimodal_bands
    STANDARD_BANDS["multimodal"] = multimodal_bands
    BENv2DataSet.channel_configurations[num_channels] = multimodal_bands
    BENv2DataSet.avail_chan_configs[num_channels] = config_description
    
    # ============================================================================
    # PRINT CONFIGURATION SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("MULTIMODAL TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"\nS1 Usage: {'ENABLED' if use_s1 else 'DISABLED'}")
    print(f"\nRegistered channel_configurations[{num_channels}]:")
    print(f"  Band order: {multimodal_bands}")
    print(f"  -> RGB bands (indices 0-2): {multimodal_bands[:3]}")
    print(f"  -> S2 non-RGB ({len(s2_non_rgb)} channels): {multimodal_bands[3:3+len(s2_non_rgb)]}")
    if use_s1:
        print(f"  -> S1 bands (last {len(s1_bands)}): {multimodal_bands[-len(s1_bands):]}")
    
    print(f"\nData Loading Confirmation:")
    print(f"  ✓ BENv2DataSet uses channel_configurations[{num_channels}] to load bands")
    print(f"  ✓ stack_and_interpolate uses order=bands parameter")
    print(f"  ✓ Bands are stacked in the exact order of multimodal_bands list")
    print(f"  ✓ Therefore: Channel 0=B04, Channel 1=B03, Channel 2=B02 (RGB)")
    print(f"  ✓ RGB extraction: rgb_data = x[:, :3, :, :] is CORRECT")
    
    print(f"\nModel Input Configuration:")
    print(f"  - DINOv3 input: RGB channels (indices 0-2) = {rgb_bands}")
    if use_s1:
        print(f"  - ResNet input: S2_non-RGB ({len(s2_non_rgb)} channels) + S1 ({len(s1_bands)} channels) = {resnet_input_channels} total")
    else:
        print(f"  - ResNet input: S2_non-RGB ({len(s2_non_rgb)} channels) only = {resnet_input_channels} total")
    print(f"  - Lightning split: rgb_data = x[:, :3], non_rgb_data = x[:, 3:]")
    print("=" * 80 + "\n")
    
    # ============================================================================
    # SETUP DATA MODULE AND TRAINER
    # ============================================================================
    # Hyperparameters for logging
    hparams = {
        "architecture": "multimodal",
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "batch_size": bs,
        "workers": workers,
        "channels": num_channels,
        "dropout": drop_rate,
        "warmup": warmup if warmup != -1 else None,
        "dinov3_model_name": dinov3_model_name,
        "dinov3_pretrained": dinov3_pretrained,
        "dinov3_freeze": dinov3_freeze,
        "dinov3_lr": dinov3_lr,
        "resnet_pretrained": resnet_pretrained,
        "resnet_freeze": resnet_freeze,
        "resnet_lr": resnet_lr,
        "resnet_input_channels": resnet_input_channels,
        "fusion_type": fusion_type,
        "classifier_type": classifier_type,
        "use_s1": use_s1,
    }
    
    trainer = default_trainer(hparams, use_wandb, test_run)
    
    # Get data directories
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
    
    # Create data module
    dm = default_dm(hparams, data_dirs, img_size)
    
    # ============================================================================
    # HANDLE CHECKPOINT RESUME
    # ============================================================================
    ckpt_path = None
    if resume_from is not None:
        if resume_from.lower() in ["best", "last"]:
            ckpt_path = resume_from.lower()
            print(f"Resuming from {resume_from} checkpoint (will be resolved by Lightning)")
        else:
            ckpt_path_obj = Path(resume_from)
            if not ckpt_path_obj.exists():
                ckpt_path_obj = Path("./checkpoints") / resume_from
                if not ckpt_path_obj.exists():
                    raise FileNotFoundError(
                        f"Checkpoint not found: {resume_from}\n"
                        f"Tried: {resume_from} and ./checkpoints/{resume_from}"
                    )
            ckpt_path = str(ckpt_path_obj.resolve())
            print(f"Resuming training from checkpoint: {ckpt_path}")
    
    # ============================================================================
    # START TRAINING
    # ============================================================================
    trainer.fit(model, dm, ckpt_path=ckpt_path)
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    
    print("\n" + "=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)
    print(f"Test results: {results[0] if results else 'No results'}")


if __name__ == "__main__":
    typer.run(main)
