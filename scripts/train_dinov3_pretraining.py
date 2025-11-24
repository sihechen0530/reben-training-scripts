"""
DINOv3 Domain-Adaptive Pre-training (DAPT) script for BigEarthNetv2.

This script performs unsupervised pre-training of DINOv3 on BigEarthNetv2
using self-supervised learning with multi-crop augmentation.

Key features:
- Weight Inflation: Adapts 3-channel DINOv3 to 14 channels (S1+S2)
- Multi-Crop Augmentation: 2 global crops (80%) + 8 local crops (30%)
- Student-Teacher Architecture: Momentum updates and centering
- Native Resolution: Trains at 128x128 (no upscaling)
- Unsupervised: No labels required
"""

import sys
from pathlib import Path

# Add parent directory to path to allow importing reben_publication
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import lightning.pytorch as pl
import torch
import typer
from typing import Optional
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BENv2_utils import resolve_data_dir, STANDARD_BANDS
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet

from reben_publication.DINOv3_Pretraining import DINOv3Pretraining
from scripts.utils import (
    get_benv2_dir_dict,
    default_trainer,
    get_job_run_directory,
    snapshot_config_file,
)
import lightning.pytorch as pl
from pathlib import Path

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def create_pretraining_trainer(
    hparams: dict,
    use_wandb: bool,
    test_run: bool,
    devices: Optional[int] = None,
    strategy: Optional[str] = None,
):
    """
    Create a PyTorch Lightning Trainer for unsupervised pre-training.
    
    This trainer monitors training loss instead of validation metrics.
    """
    # Setup logger
    if use_wandb:
        logger = pl.loggers.WandbLogger(project="BENv2-DAPT", log_model=True)
    else:
        logger = pl.loggers.WandbLogger(project="BENv2-DAPT", log_model=False, mode="disabled")
    
    logger.log_hyperparams(hparams)
    
    # Setup checkpoint directory
    run_name = hparams.get('run_name', 'default')
    ckpt_root = get_job_run_directory(run_name)
    dirpath = ckpt_root / "checkpoints"
    dirpath.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint callback - monitor training loss (for pre-training)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train/loss",
        dirpath=str(dirpath.resolve()),
        filename=f"{hparams['architecture']}-{hparams['seed']}-{hparams['channels']}-{run_name}-train_loss-" + "{train/loss:.4f}",
        save_top_k=1,
        mode="min",  # Lower loss is better
        auto_insert_metric_name=False,
        enable_version_counter=False,
        save_last=True,  # Also save last checkpoint
    )
    
    # Learning rate monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    
    # Trainer configuration
    trainer_kwargs = {
        "max_epochs": 4 if test_run else hparams["epochs"],
        "limit_train_batches": 4 if test_run else None,
        "logger": logger,
        "accelerator": "auto",
        "callbacks": [checkpoint_callback, lr_monitor],
        "enable_progress_bar": True,
        "enable_model_summary": True,
    }
    
    # Add devices if specified
    if devices is not None:
        trainer_kwargs["devices"] = devices
        # Auto-select strategy if not specified and using multiple devices
        if strategy is None and devices > 1:
            strategy = "ddp"
    
    # Add strategy if specified
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    
    trainer = pl.Trainer(**trainer_kwargs)
    return trainer


def get_bands(bandconfig: str):
    """
    Get bands and number of channels for a given band configuration.
    
    Uses existing STANDARD_BANDS configurations where possible to ensure
    compatibility with BENv2DataModule.
    
    Args:
        bandconfig: Band configuration string
        
    Returns:
        Tuple of (bands list, number of channels)
    """
    from configilm.extra.BENv2_utils import STANDARD_BANDS
    
    bandconfig_lower = bandconfig.lower()
    
    if bandconfig_lower == "all":
        # Use existing STANDARD_BANDS[12]: 10m + 20m Sentinel-2 + 10m Sentinel-1
        bands = STANDARD_BANDS[12]
        channels = 12
    elif bandconfig_lower == "s1_s2" or bandconfig_lower == "all_14ch":
        # Custom 14-channel: S1 (2) + S2 all bands (12) = 14 channels
        # This includes all S2 bands including B01
        bands = ["VV", "VH"] + ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
        channels = 14
    elif bandconfig_lower == "s1_s2_no_rgb" or bandconfig_lower == "all_no_rgb":
        # Use existing STANDARD_BANDS["all_no_rgb"]: S1 + S2 without RGB (11 channels)
        bands = STANDARD_BANDS["all_no_rgb"]  # This is registered as 11 channels
        channels = 11
    elif bandconfig_lower == "s2":
        # Use existing STANDARD_BANDS[10]: 10m + 20m Sentinel-2
        bands = STANDARD_BANDS[10]
        channels = 10
    elif bandconfig_lower == "s2_no_rgb":
        # Use existing STANDARD_BANDS["s2_no_rgb"]: S2 without RGB (9 channels)
        bands = STANDARD_BANDS["s2_no_rgb"]  # This is registered as 9 channels
        channels = 9
    elif bandconfig_lower == "s1":
        # Use existing STANDARD_BANDS[2]: Sentinel-1
        bands = STANDARD_BANDS[2]
        channels = 2
    elif bandconfig_lower == "rgb":
        # RGB only: 3 channels
        bands = ["B04", "B03", "B02"]
        channels = 3
    else:
        raise ValueError(
            f"Unknown band configuration: {bandconfig}. "
            f"Choose from: all (12ch), all_14ch (14ch), s1_s2_no_rgb/all_no_rgb (11ch), "
            f"s2 (10ch), s2_no_rgb (9ch), s1 (2ch), rgb (3ch)"
        )
    
    return bands, channels


def create_datamodule(hparams, data_dirs, img_size: int = 128):
    """
    Create data module for pre-training (unsupervised, no labels needed).
    
    Optimized for maximum throughput:
    - Auto-detects CPU cores if workers not specified
    - Uses pin_memory=True for faster CPU-to-GPU transfer
    - Uses persistent_workers=True to avoid worker restart overhead
    
    Args:
        hparams: Hyperparameters dictionary
        data_dirs: Data directory dictionary
        img_size: Image size (128 for native resolution)
        
    Returns:
        BENv2DataModule instance
    """
    import os
    import multiprocessing
    from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
    from configilm.extra.BENv2_utils import STANDARD_BANDS
    from torchvision import transforms
    
    # Get bands and channels
    bands, channels = get_bands(hparams["bandconfig"])
    
    # Register band configuration in BENv2DataSet if not already registered
    # This is required for the datamodule to know which bands to load
    if channels not in BENv2DataSet.channel_configurations:
        BENv2DataSet.channel_configurations[channels] = bands
        BENv2DataSet.avail_chan_configs[channels] = f"DAPT: {hparams['bandconfig']}"
        # Also register in STANDARD_BANDS for consistency
        STANDARD_BANDS[channels] = bands
        print(f"Registered channel_configurations[{channels}] with {len(bands)} bands: {bands}")
    
    # Auto-detect workers if not specified (use physical cores for maximum throughput)
    num_workers = hparams.get("workers")
    if num_workers is None:
        # Use physical CPU cores (not logical/hyperthreaded cores)
        try:
            num_workers = len(os.sched_getaffinity(0))  # Linux: get CPU affinity
        except AttributeError:
            # Fallback for non-Linux: use CPU count
            num_workers = multiprocessing.cpu_count()
        print(f"Auto-detected {num_workers} CPU cores for data loading")
    else:
        num_workers = hparams["workers"]
    
    # Create data module
    # The datamodule will use channels to look up the band configuration
    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=hparams["batch_size"],
        num_workers_dataloader=num_workers,
        img_size=(channels, img_size, img_size),
    )
    
    # Optimize DataLoader settings for maximum throughput
    # Override train_dataloader method to add pin_memory and persistent_workers
    from torch.utils.data import DataLoader
    
    original_train_dataloader = dm.train_dataloader
    
    def optimized_train_dataloader():
        """Optimized train dataloader with pin_memory and persistent_workers."""
        # Get the original dataloader
        original_loader = original_train_dataloader()
        
        # Create a new DataLoader with optimized settings
        # We need to recreate it because DataLoader properties are set at creation
        optimized_loader = DataLoader(
            original_loader.dataset,
            batch_size=original_loader.batch_size,
            shuffle=original_loader.sampler is None,  # shuffle if no custom sampler
            sampler=original_loader.sampler,
            num_workers=num_workers,
            pin_memory=True,  # Faster CPU-to-GPU transfer
            persistent_workers=num_workers > 0,  # Keep workers alive between epochs
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
            drop_last=original_loader.drop_last,
        )
        return optimized_loader
    
    dm.train_dataloader = optimized_train_dataloader
    
    # For pre-training, we don't need normalization transforms
    # We'll apply normalization in the augmentation pipeline
    # But we need to ensure images are in [0, 1] range
    # The default transforms might include normalization, so we'll replace them
    # with minimal transforms that just ensure proper format
    
    # Simple transform: just ensure tensor format
    # Images from BENv2DataModule should already be in [0, 1] range
    train_transforms = transforms.Compose([
        # No normalization needed - we'll handle it in augmentation
        # Just ensure it's a tensor (should already be)
    ])
    dm.train_transform = train_transforms
    
    # Print optimization settings
    print(f"\n{'='*60}")
    print("Data Loading Optimizations:")
    print(f"{'='*60}")
    print(f"  Workers: {num_workers} (maximize for H200: 16-32)")
    print(f"  Pin Memory: True (faster CPU-to-GPU transfer)")
    print(f"  Persistent Workers: {num_workers > 0} (avoid worker restart overhead)")
    print(f"  Prefetch Factor: 2 (prefetch batches in workers)")
    print(f"  LMDB: Using LMDB for fast I/O (already configured)")
    print(f"\n  Note: For >256GB RAM, consider pre-loading LMDB to /dev/shm")
    print(f"        This can be done by symlinking the LMDB directory to /dev/shm")
    print(f"{'='*60}\n")
    
    return dm


def main(
    dinov3_model_name: str = typer.Option(
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="DINOv3 HuggingFace model name"
    ),
    bandconfig: str = typer.Option(
        "all",
        help="Band configuration: all (12ch S1+S2), all_14ch (14ch S1+S2), all_no_rgb (11ch), s2 (10ch), s2_no_rgb (9ch), s1 (2ch), rgb (3ch)"
    ),
    seed: int = typer.Option(42, help="Random seed"),
    lr: float = typer.Option(5e-6, help="Learning rate (low for DAPT)"),
    epochs: int = typer.Option(100, help="Number of epochs"),
    bs: int = typer.Option(64, help="Batch size (smaller due to multi-crop)"),
    weight_decay: float = typer.Option(0.04, help="Weight decay"),
    warmup_epochs: int = typer.Option(10, help="Warmup epochs"),
    workers: int = typer.Option(
        None,
        help="Number of workers (None = auto-detect CPU cores for maximum throughput). "
             "For H200, use max physical cores (16-32)."
    ),
    image_size: int = typer.Option(128, help="Image size (native resolution, no upscaling)"),
    global_crop_size: int = typer.Option(128, help="Global crop size"),
    local_crop_size: int = typer.Option(96, help="Local crop size"),
    num_local_crops: int = typer.Option(8, help="Number of local crops"),
    num_global_crops: int = typer.Option(2, help="Number of global crops"),
    out_dim: int = typer.Option(
        8192,
        help="Output dimension (number of prototypes). Default: 8192 for small models. "
             "Use 16384 for base, 65536 for large models. Larger values increase parameters significantly."
    ),
    momentum_teacher: float = typer.Option(0.996, help="Momentum for teacher updates"),
    center_momentum: float = typer.Option(0.9, help="Momentum for centering"),
    warmup_teacher_temp: float = typer.Option(0.04, help="Initial teacher temperature"),
    teacher_temp: float = typer.Option(0.04, help="Final teacher temperature"),
    warmup_teacher_temp_epochs: int = typer.Option(30, help="Epochs to warmup teacher temperature"),
    use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
    test_run: bool = typer.Option(False, help="Run training with fewer epochs and batches"),
    resume_from: str = typer.Option(
        None,
        help="Path to checkpoint file to resume training from"
    ),
    config_path: str = typer.Option(
        None,
        help="Path to config YAML file for data directory configuration"
    ),
    run_name: str = typer.Option(
        None,
        help="Custom name for this run"
    ),
    devices: int = typer.Option(
        None,
        help="Number of GPUs to use (None = auto-detect)"
    ),
    strategy: str = typer.Option(
        None,
        help="Training strategy (None = auto, 'ddp', 'ddp_spawn', etc.)"
    ),
):
    """
    Main training function for DINOv3 Domain-Adaptive Pre-training.
    
    This script performs unsupervised pre-training of DINOv3 on BigEarthNetv2
    using self-supervised learning. The model learns to extract features from
    multi-channel satellite imagery without using labels.
    """
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory."
    
    # Set seed
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    # Get bands and channels
    bands, channels = get_bands(bandconfig)
    print(f"\n{'='*60}")
    print(f"DINOv3 Domain-Adaptive Pre-training Configuration")
    print(f"{'='*60}")
    print(f"Model: {dinov3_model_name}")
    print(f"Band config: {bandconfig}")
    print(f"Bands: {bands}")
    print(f"Channels: {channels}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Global crops: {num_global_crops} x {global_crop_size}x{global_crop_size}")
    print(f"Local crops: {num_local_crops} x {local_crop_size}x{local_crop_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {bs}")
    print(f"{'='*60}\n")
    
    # Generate unique run name if not provided
    if run_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"dinov3_pretrain_{bandconfig}_{channels}ch_{seed}_{timestamp}"
    
    # Create run directory and snapshot config
    run_dir = get_job_run_directory(run_name)
    copied_config = snapshot_config_file(config_path, run_dir)
    if copied_config:
        print(f"[Config Snapshot] Copied {copied_config} into run directory for reproducibility.")
    
    # Extract architecture name from dinov3_model_name for checkpoint naming
    # e.g., "facebook/dinov3-vitb16-pretrain-lvd1689m" -> "dinov3-vitb16"
    if "/" in dinov3_model_name:
        architecture_short = dinov3_model_name.split("/")[-1].split("-pretrain")[0]
    else:
        architecture_short = dinov3_model_name.split("-pretrain")[0]
    
    # Auto-detect workers if not specified
    if workers is None:
        import multiprocessing
        import os
        try:
            workers = len(os.sched_getaffinity(0))  # Linux: get CPU affinity
        except AttributeError:
            workers = multiprocessing.cpu_count()
        print(f"Auto-detected {workers} CPU cores for data loading")
    
    # Hyperparameters
    hparams = {
        "architecture": architecture_short,  # Required by default_trainer
        "dinov3_model_name": dinov3_model_name,
        "bandconfig": bandconfig,
        "channels": channels,
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "batch_size": bs,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "workers": workers,
        "image_size": image_size,
        "global_crop_size": global_crop_size,
        "local_crop_size": local_crop_size,
        "num_local_crops": num_local_crops,
        "num_global_crops": num_global_crops,
        "out_dim": out_dim,
        "momentum_teacher": momentum_teacher,
        "center_momentum": center_momentum,
        "warmup_teacher_temp": warmup_teacher_temp,
        "teacher_temp": teacher_temp,
        "warmup_teacher_temp_epochs": warmup_teacher_temp_epochs,
        "run_name": run_name,
    }
    
    # Create model
    print("Creating DINOv3 pre-training model...")
    model = DINOv3Pretraining(
        dinov3_model_name=dinov3_model_name,
        num_input_channels=channels,
        image_size=image_size,
        out_dim=out_dim,
        warmup_teacher_temp=warmup_teacher_temp,
        teacher_temp=teacher_temp,
        warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
        num_local_crops=num_local_crops,
        num_global_crops=num_global_crops,
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        momentum_teacher=momentum_teacher,
        center_momentum=center_momentum,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
    )
    
    # Create trainer for pre-training (monitors training loss, not validation metrics)
    trainer = create_pretraining_trainer(
        hparams,
        use_wandb,
        test_run,
        devices=devices,
        strategy=strategy,
    )
    
    # Get data directories
    hostname, data_dirs = get_benv2_dir_dict(config_path=config_path)
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
    
    # Create data module
    print("Creating data module...")
    dm = create_datamodule(hparams, data_dirs, img_size=image_size)
    
    # Handle checkpoint resume
    ckpt_path = None
    if resume_from is not None:
        ckpt_path_obj = Path(resume_from)
        if not ckpt_path_obj.exists():
            # Try relative to checkpoints directory
            ckpt_path_obj = Path("./checkpoints") / resume_from
            if not ckpt_path_obj.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {resume_from}\n"
                    f"Tried: {resume_from} and ./checkpoints/{resume_from}"
                )
        ckpt_path = str(ckpt_path_obj.resolve())
        print(f"Resuming training from checkpoint: {ckpt_path}")
    
    # Train
    print("\nStarting pre-training...")
    trainer.fit(model, dm, ckpt_path=ckpt_path)
    
    # Save model
    print("\nSaving pre-trained model...")
    model_name = f"dinov3_pretrained_{bandconfig}_{channels}ch"
    save_dir = Path("hf_models") / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save student model (the one we'll use for fine-tuning)
    torch.save({
        "model_state_dict": model.student.state_dict(),
        "config": {
            "dinov3_model_name": dinov3_model_name,
            "num_input_channels": channels,
            "image_size": image_size,
            "embed_dim": model.student.embed_dim,
        },
        "hparams": hparams,
    }, save_dir / "student_model.pt")
    
    print(f"Model saved to: {save_dir}")
    print("\n=== Pre-training finished ===")
    print(f"To use this model for fine-tuning, load the student model:")
    print(f"  checkpoint = torch.load('{save_dir / 'student_model.pt'}')")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")


if __name__ == "__main__":
    typer.run(main)


