"""
Training script for multimodal classification model.

This script trains a multimodal model that combines:
- S2 RGB data (3 channels) through DINOv3 backbone
- S2 non-RGB (11 channels) + S1 (2 channels) through ResNet101 backbone
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
from configilm.extra.BENv2_utils import resolve_data_dir

from multimodal.lightning_module import MultiModalLightningModule
from scripts.utils import get_benv2_dir_dict, default_trainer, default_dm

__author__ = "BIFOLD/RSiM TU Berlin"


def main(
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(32, help="Batch size"),
        drop_rate: float = typer.Option(0.15, help="Dropout rate"),
        warmup: int = typer.Option(1000, help="Warmup steps, set to -1 for automatic calculation"),
        workers: int = typer.Option(8, help="Number of workers"),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        test_run: bool = typer.Option(True, help="Run training with fewer epochs and batches"),
        # DINOv3 configuration
        dinov3_model_name: str = typer.Option("facebook/dinov3-vitb16-pretrain-lvd1689m", 
                                              help="DINOv3 HuggingFace model name"),
        dinov3_pretrained: bool = typer.Option(True, help="Use pretrained DINOv3 weights"),
        dinov3_freeze: bool = typer.Option(False, help="Freeze DINOv3 backbone"),
        dinov3_lr: float = typer.Option(1e-4, help="Learning rate for DINOv3 backbone"),
        dinov3_checkpoint: str = typer.Option(None, 
                                             help="Path to checkpoint file to load DINOv3 backbone weights from. "
                                                  "Can be absolute path or relative to scripts/checkpoints/"),
        # ResNet101 configuration
        resnet_pretrained: bool = typer.Option(True, help="Use pretrained ResNet101 weights"),
        resnet_freeze: bool = typer.Option(False, help="Freeze ResNet101 backbone"),
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
        # Training configuration
        resume_from: str = typer.Option(None, help="Path to checkpoint file to resume training from. "
                                                   "Can be a full path or 'best'/'last' to use the best/last checkpoint from the checkpoint directory."),
):
    """
    Train a multimodal classification model.
    
    The model uses:
    - DINOv3 backbone for S2 RGB data (3 channels)
    - ResNet101 backbone for S2 non-RGB (11 channels) + S1 (2 channels)
    - Late fusion to combine features
    - Classification head for final predictions
    """
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    
    # FIXED MODEL PARAMETERS
    num_classes = 19
    img_size = 120
    
    # set seed
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    # Build configuration
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
        # RGB channels will be set after band configuration
    }
    
    if fusion_type == "linear_projection" and fusion_output_dim is not None:
        config["fusion"]["output_dim"] = fusion_output_dim
    
    if classifier_type == "mlp":
        config["classifier"]["hidden_dim"] = classifier_hidden_dim
    
    # Resolve checkpoint paths
    dinov3_ckpt_path = None
    if dinov3_checkpoint is not None:
        dinov3_ckpt_path_obj = Path(dinov3_checkpoint)
        if not dinov3_ckpt_path_obj.exists():
            # Try relative to checkpoints directory
            dinov3_ckpt_path_obj = Path("./checkpoints") / dinov3_checkpoint
            if not dinov3_ckpt_path_obj.exists():
                raise FileNotFoundError(
                    f"DINOv3 checkpoint not found: {dinov3_checkpoint}\n"
                    f"Tried: {dinov3_checkpoint} and ./checkpoints/{dinov3_checkpoint}"
                )
        dinov3_ckpt_path = str(dinov3_ckpt_path_obj.resolve())
        print(f"Using DINOv3 checkpoint: {dinov3_ckpt_path}")
    
    resnet_ckpt_path = None
    if resnet_checkpoint is not None:
        resnet_ckpt_path_obj = Path(resnet_checkpoint)
        if not resnet_ckpt_path_obj.exists():
            # Try relative to checkpoints directory
            resnet_ckpt_path_obj = Path("./checkpoints") / resnet_checkpoint
            if not resnet_ckpt_path_obj.exists():
                raise FileNotFoundError(
                    f"ResNet checkpoint not found: {resnet_checkpoint}\n"
                    f"Tried: {resnet_checkpoint} and ./checkpoints/{resnet_checkpoint}"
                )
        resnet_ckpt_path = str(resnet_ckpt_path_obj.resolve())
        print(f"Using ResNet checkpoint: {resnet_ckpt_path}")
    
    # Create model
    model = MultiModalLightningModule(
        config=config,
        lr=lr,
        warmup=None if warmup == -1 else warmup,
        dinov3_checkpoint=dinov3_ckpt_path,
        resnet_checkpoint=resnet_ckpt_path,
        freeze_dinov3=dinov3_freeze if dinov3_ckpt_path else False,
        freeze_resnet=resnet_freeze if resnet_ckpt_path else False,
        dinov3_model_name=dinov3_model_name,  # Pass explicit model name
    )
    
    # Hyperparameters for logging (channels will be updated after band configuration)
    hparams = {
        "architecture": "multimodal",
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "batch_size": bs,
        "workers": workers,
        "channels": None,  # Will be set after band configuration
        "dropout": drop_rate,
        "warmup": warmup if warmup != -1 else None,
        "dinov3_model_name": dinov3_model_name,
        "dinov3_pretrained": dinov3_pretrained,
        "dinov3_freeze": dinov3_freeze,
        "dinov3_lr": dinov3_lr,
        "resnet_pretrained": resnet_pretrained,
        "resnet_freeze": resnet_freeze,
        "resnet_lr": resnet_lr,
        "fusion_type": fusion_type,
        "classifier_type": classifier_type,
    }
    
    trainer = default_trainer(hparams, use_wandb, test_run)
    
    # Get data directories
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
    
    # Create data module with all S2 bands + S1 bands
    # Use "all_full" to get all S2 bands (14 channels: 3 RGB + 11 non-RGB) + S1 (2 channels)
    from configilm.extra.BENv2_utils import STANDARD_BANDS
    from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet
    
    # Get all S2 bands (includes all resolutions, should be 14 channels)
    # S2 full includes: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12, B13
    # But standard S2 has 12-13 bands. Let's use S2_full which should have all bands
    s2_bands = STANDARD_BANDS.get("S2", STANDARD_BANDS.get("s2_full", []))
    s1_bands = STANDARD_BANDS.get("S1", [])
    
    # If S2 doesn't have 14 channels, we'll use what's available
    # The model expects: S2 (14 channels: 3 RGB + 11 non-RGB) + S1 (2 channels)
    # RGB channels should be: B04 (Red), B03 (Green), B02 (Blue)
    # So we need to ensure B02, B03, B04 are the first 3 channels
    
    # Define RGB bands and reorder S2 bands to put RGB first
    # This ensures that when data is loaded, RGB channels will be at the front
    rgb_bands = ["B04", "B03", "B02"]  # RGB bands for DINOv3
    s2_non_rgb = [b for b in s2_bands if b not in rgb_bands]
    s2_ordered = rgb_bands + s2_non_rgb  # RGB first, then other S2 bands
    
    # Combine: S2 (ordered: RGB + non-RGB) + S1
    # Final order: [RGB (3), S2_non-RGB (N), S1 (2)]
    multimodal_bands = s2_ordered + s1_bands
    num_channels = len(multimodal_bands)
    
    # Register custom configuration
    # IMPORTANT ASSUMPTION: We assume that BENv2DataSet loads bands in the order specified
    # in channel_configurations[num_channels]. The configilm library's stack_and_interpolate
    # function uses an 'order' parameter, suggesting bands are stacked in list order.
    # 
    # However, we cannot be 100% certain without checking the BENv2DataSet source code.
    # If the dataloader does NOT respect this order, RGB channels may not be at [0,1,2].
    # 
    # By registering multimodal_bands with RGB first, we ensure that IF the dataloader
    # respects the order, RGB will be at indices [0, 1, 2].
    STANDARD_BANDS[num_channels] = multimodal_bands
    STANDARD_BANDS["multimodal"] = multimodal_bands
    BENv2DataSet.channel_configurations[num_channels] = multimodal_bands
    BENv2DataSet.avail_chan_configs[num_channels] = "Multimodal (S2 ordered + S1)"
    
    # Print registered configuration for verification
    print(f"\nRegistered channel_configurations[{num_channels}]:")
    print(f"  Band order: {multimodal_bands}")
    print(f"  -> RGB bands (indices 0-2): {multimodal_bands[:3]}")
    print(f"  -> S2 non-RGB: {multimodal_bands[3:len(s2_ordered)]}")
    print(f"  -> S1 bands (last 2): {multimodal_bands[-2:]}")
    print(f"\n  NOTE: This assumes BENv2DataSet loads bands in this exact order.")
    print(f"        If unsure, verify by checking configilm library source code or")
    print(f"        running scripts/verify_band_order.py to test.\n")
    
    # Configure RGB channels in model config (for reference/documentation)
    # Note: The Lightning module now directly separates RGB and non-RGB+S1 data,
    # so the model doesn't need to know RGB channel indices. We keep this config
    # for documentation and reference purposes.
    config["rgb_channels"] = [0, 1, 2]  # RGB channels are at indices [0, 1, 2]
    config["rgb_band_names"] = rgb_bands  # Store band names for reference
    config["s2_band_order"] = s2_ordered  # Store S2 band order for reference
    config["s1_band_order"] = s1_bands  # Store S1 band order for reference
    
    # Update hparams
    hparams["channels"] = num_channels
    
    print(f"Using {num_channels} channels: {len(s2_ordered)} S2 + {len(s1_bands)} S1")
    print(f"Channel order: RGB={rgb_bands} (indices 0-2) + S2_non-RGB ({len(s2_ordered)-3} channels) + S1={s1_bands} (last 2 channels)")
    print(f"Model input configuration:")
    print(f"  - DINOv3 input: RGB channels (indices 0-2) = {rgb_bands}")
    print(f"  - ResNet input: S2_non-RGB ({len(s2_ordered)-3} channels) + S1 ({len(s1_bands)} channels)")
    print(f"  Note: Lightning module will split data as: rgb_data=x[:,:3] and non_rgb_s1_data=x[:,3:]")
    
    # Create data module
    dm = default_dm(hparams, data_dirs, img_size)
    
    # Handle checkpoint resume
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
    
    trainer.fit(model, dm, ckpt_path=ckpt_path)
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    
    print("=== Training finished ===")
    print(f"Test results: {results[0] if results else 'No results'}")


if __name__ == "__main__":
    typer.run(main)

