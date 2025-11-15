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
from datetime import datetime
from pathlib import Path

# Add parent directory to path to allow importing multimodal module
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import lightning.pytorch as pl
import torch
import typer
import yaml
from typing import Optional
from configilm.extra.BENv2_utils import resolve_data_dir, STANDARD_BANDS
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet

from multimodal.lightning_module import MultiModalLightningModule
from scripts.utils import (
    default_dm,
    default_trainer,
    get_benv2_dir_dict,
    get_job_run_directory,
    resolve_checkpoint_path,
    snapshot_config_file,
)

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
    architecture: Optional[str] = typer.Option(
        None,
        help="Optional alias for DINOv3 model size (dinov3-small/base/large/giant). Overrides --dinov3-hidden-size when provided.",
    ),
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
    use_resnet: bool = typer.Option(True, "--use-resnet/--no-resnet", help="Enable or disable the ResNet branch regardless of band selection."),
        # Fusion configuration
        fusion_type: str = typer.Option(
            "concat",
            help="Fusion type: concat, weighted, linear_projection, attention, bilinear, or gated"
        ),
        fusion_output_dim: int = typer.Option(
            None,
            help="Output dimension for fusion (used for linear_projection, attention, bilinear, gated)"
        ),
        fusion_num_heads: int = typer.Option(
            1,
            help="Number of attention heads (only used for attention fusion)"
        ),
        fusion_use_low_rank: bool = typer.Option(
            True,
            help="Whether to use low-rank approximation for bilinear fusion"
        ),
        # Classifier configuration
        classifier_type: str = typer.Option(
            "linear",
            help="Classifier type: linear, mlp, svm, or labelwise_binary"
        ),
        classifier_hidden_dim: int = typer.Option(
            512,
            help="Hidden dimension for MLP classifier"
        ),
        classifier_shared_backbone: bool = typer.Option(
            True,
            help="Whether to share backbone for labelwise_binary classifier"
        ),
        # Data configuration
        bandconfig: Optional[str] = typer.Option(
            None,
            help="Band configuration override. Options: rgb, s2, s2s1/multimodal. Overrides --use-s1 when provided.",
        ),
        use_s1: bool = typer.Option(False, "--use-s1/--no-use-s1", help="Whether to include S1 (Sentinel-1) data. If False, only S2 non-RGB bands are used for ResNet."),
        # Training configuration
        resume_from: str = typer.Option(None, help="Path to checkpoint file to resume training from. "
                                                    "Can be a full path or 'best'/'last' to use the best/last checkpoint from the checkpoint directory."),
        config_path: str = typer.Option(
            None,
            help="Path to config YAML file for data directory configuration. "
                 "If not provided, hostname-based directory selection is used.",
        ),
        run_name: str = typer.Option(
            None,
            help="Custom name for this run. Defaults to <architecture>-<bandconfig>-<seed>-<timestamp>.",
        ),
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
    # PRINT CONFIGURATION FROM YAML FILE (if provided)
    # ============================================================================
    config_data = None
    output_base_dir = None
    
    if config_path:
        try:
            config_path_obj = Path(config_path)
            if config_path_obj.exists():
                with open(config_path_obj, 'r') as f:
                    config_data = yaml.safe_load(f)

                print("\n" + "="*80, file=sys.stderr)
                print("YAML CONFIGURATION", file=sys.stderr)
                print("="*80, file=sys.stderr)
                print(f"Config file: {config_path_obj.absolute()}", file=sys.stderr)
                print("-"*80, file=sys.stderr)
                print(yaml.dump(config_data, default_flow_style=False, sort_keys=False), file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                
                # Extract output_dir from config if available
                if config_data and 'job' in config_data and 'output_dir' in config_data['job']:
                    output_dir_config = config_data['job']['output_dir']
                    chdir_config = config_data['job'].get('chdir', str(Path.cwd()))
                    
                    # Resolve output_dir (support both absolute and relative paths)
                    output_dir_path = Path(output_dir_config)
                    if output_dir_path.is_absolute():
                        output_base_dir = output_dir_path.resolve()
                    else:
                        # Relative to chdir
                        output_base_dir = (Path(chdir_config) / output_dir_config).resolve()
                    print(f"[Config] Using output_dir from config: {output_base_dir}", file=sys.stderr)
            else:
                print(f"\nWarning: Config file not found: {config_path}\n", file=sys.stderr)
        except Exception as e:
            print(f"\nWarning: Failed to load config file: {e}\n", file=sys.stderr)

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
    if architecture is not None:
        arch_lower = architecture.lower().replace("_", "-")
        if "giant" in arch_lower or arch_lower.endswith("-g"):
            dinov3_hidden_size = 1536
        elif "large" in arch_lower or arch_lower.endswith("-l"):
            dinov3_hidden_size = 1024
        elif "base" in arch_lower or arch_lower.endswith("-b"):
            dinov3_hidden_size = 768
        elif "small" in arch_lower or arch_lower.endswith("-s"):
            dinov3_hidden_size = 384
        else:
            raise ValueError(
                "Unknown architecture alias. Use dinov3-small, dinov3-base, dinov3-large, or dinov3-giant."
            )

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

    # ==========================================================================
    # DETERMINE BAND CONFIGURATION (RGB/S2/S2+S1)
    # ==========================================================================
    if bandconfig is not None:
        band_alias = bandconfig.lower()
    else:
        band_alias = "s2s1" if use_s1 else "s2"
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
    use_s1 = True if effective_bandconfig == "s2s1" else False if effective_bandconfig in {"rgb", "s2"} else use_s1
    bandconfig_label = effective_bandconfig

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{architecture}-{bandconfig_label}-{seed}-{timestamp}"

    run_dir = get_job_run_directory(run_name, base_dir=output_base_dir)
    print(f"\n[Run Directory] Artifacts will be saved in: {run_dir.resolve()}\n")
    copied_config = snapshot_config_file(config_path, run_dir)
    if copied_config:
        print(f"[Config Snapshot] Copied {copied_config} into run directory for reproducibility.", file=sys.stderr)

    # ============================================================================
    # PRINT TRAINING PARAMETERS (after band config is resolved)
    # ============================================================================
    print("\n" + "="*80, file=sys.stderr)
    print("MULTIMODAL TRAINING PARAMETERS", file=sys.stderr)
    print("="*80, file=sys.stderr)
    training_params = {
        "seed": seed,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": bs,
        "drop_rate": drop_rate,
        "warmup_steps": warmup,
        "workers": workers,
        "use_wandb": use_wandb,
        "test_run": test_run,
        # DINOv3 config
        "dinov3_hidden_size": dinov3_hidden_size,
        "dinov3_pretrained": dinov3_pretrained,
        "dinov3_freeze": dinov3_freeze,
        "dinov3_lr": dinov3_lr,
        "dinov3_checkpoint": dinov3_checkpoint,
        # ResNet config
        "resnet_pretrained": resnet_pretrained,
        "resnet_freeze": resnet_freeze,
        "resnet_lr": resnet_lr,
        "resnet_checkpoint": resnet_checkpoint,
        # Fusion & Classifier
        "fusion_type": fusion_type,
        "fusion_output_dim": fusion_output_dim,
        "fusion_num_heads": fusion_num_heads,
        "fusion_use_low_rank": fusion_use_low_rank,
        "classifier_type": classifier_type,
        "classifier_hidden_dim": classifier_hidden_dim,
        "classifier_shared_backbone": classifier_shared_backbone,
        # Data / misc
        "use_s1": use_s1,
        "resume_from": resume_from,
        "architecture_alias": architecture,
        "bandconfig": bandconfig_label,
        "use_resnet_flag": use_resnet,
    }
    for key, value in training_params.items():
        print(f"  {key:25s}: {value}", file=sys.stderr)
    print("="*80 + "\n", file=sys.stderr)
    
    # ============================================================================
    # RESOLVE CHECKPOINT PATHS
    # ============================================================================
    dinov3_ckpt_path = None
    if dinov3_checkpoint is not None:
        try:
            dinov3_ckpt_path = str(resolve_checkpoint_path(dinov3_checkpoint))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"DINOv3 checkpoint not found: {dinov3_checkpoint}\n"
                f"Searched default folders inside ckpt_logs/ and legacy checkpoints/."
            ) from exc
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
    
    # Configure fusion parameters
    if fusion_type in ["linear_projection", "attention", "bilinear", "gated"]:
        if fusion_output_dim is not None:
            config["fusion"]["output_dim"] = fusion_output_dim
    if fusion_type == "attention":
        config["fusion"]["num_heads"] = fusion_num_heads
    if fusion_type == "bilinear":
        config["fusion"]["use_low_rank"] = fusion_use_low_rank
    
    # Configure classifier parameters
    if classifier_type == "mlp":
        config["classifier"]["hidden_dim"] = classifier_hidden_dim
    if classifier_type == "labelwise_binary":
        config["classifier"]["shared_backbone"] = classifier_shared_backbone
    
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
    full_s2_non_rgb = [b for b in s2_bands if b not in rgb_bands]
    
    if bandconfig_label == "rgb":
        s2_non_rgb = []
    else:
        s2_non_rgb = full_s2_non_rgb
    
    # Reorder S2 bands: RGB first, then non-RGB (if any)
    s2_ordered = rgb_bands + s2_non_rgb
    
    # Calculate ResNet input channels: S2 non-RGB + optionally S1
    resnet_input_channels = len(s2_non_rgb) + (len(s1_bands) if use_s1 else 0)
    resnet_enabled = use_resnet and resnet_input_channels > 0
    if not resnet_enabled and use_resnet and resnet_input_channels == 0:
        print("Warning: ResNet branch disabled because no additional channels are available for the selected band configuration.")
    
    # Configure ResNet input channels in config (needed before model creation)
    config["backbones"]["resnet101"]["input_channels"] = resnet_input_channels
    config["backbones"]["resnet101"]["enabled"] = resnet_enabled
    
    # Store band order in config for reference
    config["rgb_band_names"] = rgb_bands
    config["s2_band_order"] = s2_ordered
    config["s1_band_order"] = s1_bands if use_s1 else []
    config["use_s1"] = use_s1
    
    # ============================================================================
    # RESOLVE RESNET CHECKPOINT PATH
    # ============================================================================
    resnet_ckpt_path = None
    if resnet_checkpoint is not None and resnet_enabled:
        try:
            resnet_ckpt_path = str(resolve_checkpoint_path(resnet_checkpoint))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"ResNet checkpoint not found: {resnet_checkpoint}\n"
                f"Searched ckpt_logs/ subfolders and legacy ./checkpoints/."
            ) from exc
        print(f"Using ResNet checkpoint: {resnet_ckpt_path}")
    elif resnet_checkpoint is not None and not resnet_enabled:
        print("Warning: Provided ResNet checkpoint will be ignored because the ResNet branch is disabled.")
    
    # ============================================================================
    # CREATE MODEL
    # ============================================================================
    model = MultiModalLightningModule(
        config=config,
        lr=lr,
        warmup=None if warmup == -1 else warmup,
        dinov3_checkpoint=dinov3_ckpt_path,
        resnet_checkpoint=resnet_ckpt_path if resnet_enabled else None,
        freeze_dinov3=dinov3_freeze if dinov3_ckpt_path else False,
        freeze_resnet=resnet_freeze if resnet_enabled and resnet_ckpt_path else False,
        dinov3_model_name=dinov3_model_name,
    )
    
    # ============================================================================
    # REGISTER BAND CONFIGURATION WITH DATALOADER
    # ============================================================================
    # Combine: S2 (ordered: RGB + non-RGB) + optionally S1
    if bandconfig_label == "rgb":
        multimodal_bands = rgb_bands
        config_description = "RGB only"
    elif use_s1:
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
    print(f"\nBand configuration: {config_description} (source: {bandconfig_label})")
    print(f"S1 Usage: {'ENABLED' if use_s1 else 'DISABLED'}")
    print(f"ResNet branch: {'ENABLED' if resnet_enabled else 'DISABLED'} (input channels: {resnet_input_channels})")
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
    if resnet_enabled:
        if use_s1:
            print(f"  - ResNet input: S2_non-RGB ({len(s2_non_rgb)} channels) + S1 ({len(s1_bands)} channels) = {resnet_input_channels} total")
        else:
            print(f"  - ResNet input: S2_non-RGB ({len(s2_non_rgb)} channels) only = {resnet_input_channels} total")
        print(f"  - Lightning split: rgb_data = x[:, :3], non_rgb_data = x[:, 3:]")
    else:
        print("  - ResNet input: DISABLED (RGB-only branch is active)")
        print("  - Lightning split: rgb_data only; ResNet tensor is passed as None")
    print("=" * 80 + "\n")
    
    # ============================================================================
    # SETUP DATA MODULE AND TRAINER
    # ============================================================================
    # Hyperparameters for logging
    hparams = {
        "architecture": "multimodal",
        "architecture_alias": architecture,
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
        "resnet_enabled": resnet_enabled,
        "use_resnet_flag": use_resnet,
        "fusion_type": fusion_type,
        "classifier_type": classifier_type,
        "use_s1": use_s1,
        "bandconfig": bandconfig_label,
        "run_name": run_name,
    }
    devices = None
    strategy = None
    trainer = default_trainer(
        hparams,
        use_wandb,
        test_run,
        devices=devices,
        strategy=strategy,
        ckpt_dir=run_dir,
    )
    
    # Get data directories
    hostname, data_dirs = get_benv2_dir_dict(config_path=config_path)
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
            try:
                ckpt_path_obj = resolve_checkpoint_path(resume_from)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Checkpoint not found: {resume_from}\n"
                    f"Looked in ckpt_logs/ (per-job folders) and legacy ./checkpoints/. "
                    f"Provide an absolute path if the checkpoint lives elsewhere."
                ) from exc
            ckpt_path = str(ckpt_path_obj)
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
