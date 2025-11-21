import os
import socket
import shutil
import subprocess
import platform
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Optional, Union, Dict, Any

import lightning.pytorch as pl
import torch
from configilm.extra.BENv2_utils import STANDARD_BANDS
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from huggingface_hub import HfApi
from lightning.pytorch.loggers import WandbLogger
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from torchvision import transforms
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CKPT_LOG_ROOT = PROJECT_ROOT / "ckpt_logs"


_s1_bands = ["VV", "VH"]
_s2_no_rgb = ["B01", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
_s1_s2_no_rgb = _s1_bands + _s2_no_rgb

STANDARD_BANDS[11] = _s1_s2_no_rgb
STANDARD_BANDS["all_no_rgb"] = _s1_s2_no_rgb
BENv2DataSet.channel_configurations[11] = STANDARD_BANDS[11]
BENv2DataSet.avail_chan_configs[11] = "Sentinel-1 + Sentinel-2 (without RGB)"

STANDARD_BANDS[9] = _s2_no_rgb
STANDARD_BANDS["s2_no_rgb"] = _s2_no_rgb
BENv2DataSet.channel_configurations[9] = STANDARD_BANDS[9]
BENv2DataSet.avail_chan_configs[9] = "Sentinel-2 (without RGB)"


BENv2_DIR_MARS = Path("/data/kaiclasen")
BENv2_DIR_DICT_MARS = {
    "images_lmdb": BENv2_DIR_MARS / "BENv2.lmdb",
    "metadata_parquet": BENv2_DIR_MARS / "metadata.parquet",
    "metadata_snow_cloud_parquet": BENv2_DIR_MARS / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

BENv2_DIR_ERDE = Path("/faststorage/BigEarthNet-V2")
BENv2_DIR_DICT_ERDE = {
    "images_lmdb": BENv2_DIR_ERDE / "BigEarthNet-V2-LMDB",
    "metadata_parquet": BENv2_DIR_ERDE / "metadata.parquet",
    "metadata_snow_cloud_parquet": BENv2_DIR_ERDE / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

BENv2_DIR_PLUTO = Path("/data_read_only/BigEarthNet/BigEarthNet-V2/")
BENv2_DIR_DICT_PLUTO = {
    "images_lmdb": BENv2_DIR_PLUTO / "BENv2.lmdb",
    "metadata_parquet": BENv2_DIR_PLUTO / "metadata.parquet",
    "metadata_snow_cloud_parquet": BENv2_DIR_PLUTO / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

BENv2_DIR_DEFAULT = Path("/scratch/chen.sihe1")
BENv2_DIR_DICT_DEFAULT = {
    "images_lmdb": BENv2_DIR_DEFAULT / "BENv2.lmdb",
    "metadata_parquet": BENv2_DIR_DEFAULT / "metadata.parquet",
    "metadata_snow_cloud_parquet": BENv2_DIR_DEFAULT / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

BENv2_DIR_DICTS = {
    'mars': BENv2_DIR_DICT_MARS,
    'erde': BENv2_DIR_DICT_ERDE,
    'pluto': BENv2_DIR_DICT_PLUTO,
    'default': BENv2_DIR_DICT_DEFAULT,
}


def get_job_run_directory(
        run_name: Optional[str] = None,
        base_dir: Optional[Union[Path, str]] = None,
) -> Path:
    """
    Determine (and create) the run directory for current job.

    Priority:
        1. SLURM_JOB_ID (preferred for sbatch workflows)
        2. Provided run_name (if any)
        3. Timestamp-based fallback
    """
    base_path = Path(base_dir) if base_dir is not None else DEFAULT_CKPT_LOG_ROOT
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")

    if job_id:
        run_dir = base_path / job_id
    else:
        suffix = run_name or datetime.now().strftime("local_%Y%m%d_%H%M%S")
        run_dir = base_path / suffix

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config_file(config_path: Optional[str], destination_dir: Path) -> Optional[Path]:
    """
    Copy the YAML config that launched the job into the destination directory for traceability.
    """
    if not config_path:
        return None

    src = Path(config_path).expanduser()
    if not src.exists():
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)
    dst = destination_dir / src.name
    shutil.copy2(src, dst)
    return dst


def resolve_checkpoint_path(
        candidate: str,
        extra_search_dirs: Optional[List[Path]] = None,
) -> Path:
    """
    Resolve a checkpoint path by checking the provided candidate, common checkpoint roots,
    and performing a recursive search inside ckpt_logs if needed.
    """
    possible_paths = []
    candidate_path = Path(candidate).expanduser()
    possible_paths.append(candidate_path)

    default_dirs = extra_search_dirs[:] if extra_search_dirs else []
    default_dirs.extend([
        DEFAULT_CKPT_LOG_ROOT,
        SCRIPT_DIR / "ckpt_logs",
        SCRIPT_DIR / "checkpoints",
        Path("./ckpt_logs"),
        Path("./checkpoints"),
    ])

    for base in default_dirs:
        possible_paths.append(base / candidate)

    for path in possible_paths:
        if path.exists():
            return path.resolve()

    # As a last resort, search recursively inside ckpt_logs for matching filename
    for search_root in [DEFAULT_CKPT_LOG_ROOT, SCRIPT_DIR / "ckpt_logs", Path("./ckpt_logs")]:
        if search_root.exists():
            matches = list(search_root.rglob(candidate))
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        f"Checkpoint not found: {candidate}\n"
        f"Searched locations: {', '.join(str(p) for p in possible_paths)}"
    )


def get_benv2_dir_dict(config_path: Optional[str] = None) -> tuple[str, dict]:
    """
    Get the BENv2 data directories based on hostname or config file.
    
    Args:
        config_path: Optional path to config.yaml file. If provided, will read data_dir from there.
    
    Returns:
        Tuple of (hostname/source, data_dirs_dict)
    """
    hostname = socket.gethostname()
    
    # If config_path is provided, try to read from config first
    if config_path is not None:
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config and 'data' in config and 'benv2_data_dir' in config['data']:
                benv2_dir = Path(config['data']['benv2_data_dir'])
                data_dirs = {
                    "images_lmdb": benv2_dir / "BENv2.lmdb",
                    "metadata_parquet": benv2_dir / "metadata.parquet",
                    "metadata_snow_cloud_parquet": benv2_dir / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
                }
                print(f"Using data directories from config file: {config_path}")
                return "config", data_dirs
        except Exception as e:
            print(f"Warning: Could not read config from {config_path}: {e}")
            print("Falling back to hostname-based configuration")
    
    # Fallback to hostname-based configuration
    print(f"Using data directories for hostname: {hostname}")
    return hostname, BENv2_DIR_DICTS.get(hostname, BENv2_DIR_DICT_DEFAULT)


def _infere_example(S1_file: str, S2_file: str, bands: list[str], model: BigEarthNetv2_0_ImageClassifier):
    """
    Infere an example image using the given model and the specified bands.

    Note: This function is only used for generating the README.md file and should not be used for actual inference.
          There is no normalization applied to the input image, so the results might not be accurate and the individual
          loading of tiff files is not optimized for speed.
    :param S1_file: path to the Sentinel-1 image directory
    :param S2_file: path to the Sentinel-2 image directory
    :param bands: list of bands to use for the inference, e.g. ["B02", "B03", "B04"] for BGR input
    :param model: torch model to use for the inference
    :return: (results, input_rgb) where results is a dictionary with the class scores and input_rgb is the input image
    """
    import rasterio
    from configilm.extra.BENv2_utils import stack_and_interpolate, NEW_LABELS
    data = {}
    s1_files = [f for f in Path(S1_file).rglob("*.tif")]
    s2_files = [f for f in Path(S2_file).rglob("*.tiff")]
    for b in bands:
        if b in STANDARD_BANDS["S1"]:
            for f in s1_files:
                if b in f.name:
                    data[b] = f
                    break
        elif b in STANDARD_BANDS["S2"]:
            for f in s2_files:
                if b in f.name:
                    data[b] = f
                    break
    for d in data:
        # read the file
        with rasterio.open(data[d]) as src:
            img = src.read(1)
            data[d] = img

    # stack the bands to a 3D tensor based on the order of the bands
    img = stack_and_interpolate(data, order=bands, img_size=120, upsample_mode="nearest").unsqueeze(0)

    ### NOTE: At this point, the image is not normalized, so the results might not be accurate
    ### For actual inference, the data should be normalized using the same values as in the training data at this point

    # infer the image
    model.eval()
    with torch.no_grad():
        output = model(img)
    results = torch.sigmoid(output)
    results = results.squeeze().numpy()
    assert len(results) == len(NEW_LABELS), f"Expected {len(NEW_LABELS)} results, got {len(results)}"
    results = {NEW_LABELS[i]: results[i] for i in range(len(NEW_LABELS))}

    if img.shape[1] == 2:
        # S1 image, add VV/VH as third channel
        img = torch.cat([img[:, 0], img[:, 1], img[:, 0, :, :] / img[:, 1, :, :]]).unsqueeze(0)
        input_rgb = img.squeeze().numpy()[[2, 1, 0], :, :]
    elif img.shape[1] == 3:
        # RGB image, already in correct format (B04, B03, B02) = (R, G, B)
        # No reordering needed - already in RGB display order
        input_rgb = img.squeeze().numpy()
    elif img.shape[1] == 10:
        # S2 image, use only first 3 bands (B02, B03, B04) = (B, G, R)
        # Reorder to (B04, B03, B02) = (R, G, B) for RGB display
        img = img[:, :3]
        input_rgb = img.squeeze().numpy()[[2, 1, 0], :, :]
    elif img.shape[1] == 12:
        # S2 + S1 image, use only first 3 bands of S2 (B02, B03, B04) = (B, G, R)
        # Reorder to (B04, B03, B02) = (R, G, B) for RGB display
        img = img[:, :3]
        input_rgb = img.squeeze().numpy()[[2, 1, 0], :, :]
    else:
        # Fallback: assume first 3 channels and try to display
        num_channels = img.shape[1]
        img = img[:, :3] if num_channels > 3 else img
        input_rgb = img.squeeze().numpy()[[2, 1, 0], :, :] if num_channels >= 3 else img.squeeze().numpy()
    return results, input_rgb


def generate_readme(model_name: str, results: dict, hparams: dict, current_epoch: int,
                    model: BigEarthNetv2_0_ImageClassifier):
    import PIL.Image
    # read the template
    with open("README_template.md", "r") as f:
        template = f.read()
    # fill in the values
    architecture_raw, bandconfig, version = model_name.split("-")
    architecture = architecture_raw.capitalize()
    bands_used = "Sentinel-1 & Sentinel-2" if bandconfig == "all" \
        else "Sentinel-2" if bandconfig == "s2" \
        else "Sentinel-2 (RGB)" if bandconfig == "rgb" \
        else "Sentinel-1"

    template = template.replace("<BAND_CONFIG>", bandconfig)
    template = template.replace("<EPOCHS>", str(current_epoch))
    template = template.replace("<MODEL_NAME>", architecture)
    template = template.replace("<MODEL_NAME_RAW>", architecture_raw)
    template = template.replace("<DATASET_NAME>", "BigEarthNet v2.0")
    template = template.replace("<DATASET_NAME_FULL>", "BigEarthNet v2.0 (reBEN)")
    template = template.replace("<DATASET_NAME_FULL_2>", "BigEarthNet v2.0 (also known as reBEN)")
    template = template.replace("<BANDS_USED>", bands_used)
    template = template.replace("<LEARNING_RATE>", str(hparams["lr"]))
    template = template.replace("<BATCH_SIZE>", str(hparams["batch_size"]))
    template = template.replace("<DROPOUT_RATE>", str(hparams["dropout"]))
    template = template.replace("<DROP_PATH_RATE>", str(hparams["drop_path_rate"]))
    template = template.replace("<WARMUP_STEPS>", str(hparams["warmup"]) if hparams["warmup"] is not None else "10_000")
    template = template.replace("<AP_MACRO>", f"{results['test/MultilabelAveragePrecision_macro']:.6f}")
    template = template.replace("<AP_MICRO>", f"{results['test/MultilabelAveragePrecision_micro']:.6f}")
    template = template.replace("<F1_MACRO>", f"{results['test/MultilabelF1Score_macro']:.6f}")
    template = template.replace("<F1_MICRO>", f"{results['test/MultilabelF1Score_micro']:.6f}")
    template = template.replace("<PRECISION_MACRO>", f"{results['test/MultilabelPrecision_macro']:.6f}")
    template = template.replace("<PRECISION_MICRO>", f"{results['test/MultilabelPrecision_micro']:.6f}")
    template = template.replace("<SEED>", str(hparams["seed"]))

    if bandconfig == "all":
        bands = STANDARD_BANDS[12]  # 10m + 20m Sentinel-2 + 10m Sentinel-1
        vis_bands = "A Sentinel-2 image (true color representation)"
    elif bandconfig == "s2":
        bands = STANDARD_BANDS[10]  # 10m + 20m Sentinel-2
        vis_bands = "A Sentinel-2 image (true color representation)"
    elif bandconfig == "s1":
        bands = STANDARD_BANDS[2]  # Sentinel-1
        vis_bands = "A Sentinel-1 image (VV, VH and VV/VH bands are used for visualization)"
    elif bandconfig == "rgb":
        bands = ["B04", "B03", "B02"]  # RGB true color
        vis_bands = "A Sentinel-2 RGB image (true color representation using B04, B03, B02)"
    else:
        raise ValueError(f"Unsupported band configuration {bandconfig}")
    template = template.replace("<VIS_BANDS>", vis_bands)
    result, img = _infere_example("./data/S1", "./data/S2", bands, model)
    # write the example image as png
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype("uint8")
    img = img.transpose(1, 2, 0)
    img = PIL.Image.fromarray(img)
    img.save(f"hf_models/{model_name}/example.png")

    # replace results
    for i, (label, score) in enumerate(result.items()):
        template = template.replace(f"<LABEL_{i + 1}>", label)
        template = template.replace(f"<SCORE_{i + 1}>", f"{score:.6f}")

    # write the new README.md
    with open(f"hf_models/{model_name}/README.md", "w") as f:
        f.write(template)


def get_bands(bandconfig: str):
    if bandconfig == "all":
        bands = STANDARD_BANDS[12]  # 10m + 20m Sentinel-2 + 10m Sentinel-1
    elif bandconfig == "s2":
        bands = STANDARD_BANDS[10]  # 10m + 20m Sentinel-2
    elif bandconfig == "s1":
        bands = STANDARD_BANDS[2]  # Sentinel-1
    elif bandconfig == "all_full":
        bands = STANDARD_BANDS["ALL"]
    elif bandconfig == "s2_full":
        bands = STANDARD_BANDS["S2"]
    elif bandconfig == "s1_full":
        bands = STANDARD_BANDS["S1"]
    elif bandconfig == "rgb":
        # RGB true color: B04 (Red), B03 (Green), B02 (Blue)
        bands = ["B04", "B03", "B02"]
    elif bandconfig == "all_no_rgb":
        bands = STANDARD_BANDS["all_no_rgb"]
    elif bandconfig == "s2_no_rgb":
        bands = STANDARD_BANDS["s2_no_rgb"]
    else:
        raise ValueError(
            f"Unknown band configuration {bandconfig}, select one of all, s2, s1, rgb or all_full, s2_full, s1_full. The "
            f"full versions include all bands whereas the non-full versions only include the 10m & 20m bands. "
            f"rgb uses 3-channel RGB (B04, B03, B02) from Sentinel-2."
        )
    
    # Ensure RGB bands (B04, B03, B02) are always at the beginning in R, G, B order
    # This is critical for DINOv3 which expects RGB in that order
    rgb_bands = ["B04", "B03", "B02"]  # R, G, B order
    if all(band in bands for band in rgb_bands):
        # Remove RGB bands from their current positions
        other_bands = [b for b in bands if b not in rgb_bands]
        # Reorder: RGB first (R, G, B), then all other bands
        bands = rgb_bands + other_bands
    
    return bands, len(bands)


def upload_model_and_readme_to_hub(
        model: BigEarthNetv2_0_ImageClassifier,
        model_name: str,
        hf_entity: str,
        test_results: Mapping[str, float],
        hparams: dict,
        trainer: pl.Trainer,
        upload: bool):
    if upload:
        assert hf_entity is not None, "Please specify a Huggingface entity to upload the model to."
        print("=== Uploading model to Huggingface Hub ===")
        print(f"Uploading model as {model_name}")
        push_path = f"{hf_entity}/{model_name}" if hf_entity is not None else model_name
        print(f"Pushing to {push_path}")
        model.push_to_hub(push_path, commit_message=f"Upload {model_name}")
        print("=== Done ===")
    else:
        print("=== Skipping upload to Huggingface Hub ===")

    # upload new README.md to Huggingface Hub
    generate_readme(model_name, test_results, hparams, current_epoch=trainer.current_epoch, model=model)
    if upload and hf_entity is not None:
        print("=== Uploading README.md to Huggingface Hub ===")
        api = HfApi()
        print(f"Uploading README.md to {hf_entity}/{model_name}")
        api.upload_file(
            path_or_fileobj=f"hf_models/{model_name}/README.md",
            path_in_repo="README.md",
            repo_id=f"{hf_entity}/{model_name}",
        )
        print(f"Uploading example image to {hf_entity}/{model_name}")
        api.upload_file(
            path_or_fileobj=f"hf_models/{model_name}/example.png",
            path_in_repo="example.png",
            repo_id=f"{hf_entity}/{model_name}",
        )
        print("=== Done ===")
    else:
        print("=== Skipping upload of README.md to Huggingface Hub ===")
        if not upload:
            print("    Reason: Upload to Huggingface Hub was disabled.")
        elif hf_entity is None:
            print("    Reason: No Huggingface Hub entity specified.")
        else:
            print("    Reason: Unknown error.")


def default_trainer(
        hparams: dict,
        use_wandb: bool,
        test_run: bool,
        devices: Optional[int] = None,
        strategy: Optional[str] = None,
        ckpt_dir: Optional[Union[Path, str]] = None,
):
    """
    Create a PyTorch Lightning Trainer with default configuration.
    
    Args:
        hparams: Hyperparameters dictionary
        use_wandb: Whether to use wandb logging
        test_run: Whether this is a test run (limits batches)
        devices: Number of GPUs to use (None = auto-detect, int = specific number)
        strategy: Training strategy (None = auto, "ddp", "ddp_spawn", "deepspeed", etc.)
    
    Returns:
        Configured PyTorch Lightning Trainer
    """
    # we assume, that we are already logged in to wandb
    if use_wandb:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=True)
    else:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=False, mode="disabled")

    logger.log_hyperparams(hparams)

    # Include run_name in filename to prevent conflicts when running multiple trainings
    run_name = hparams.get('run_name', 'default')
    ckpt_root = Path(ckpt_dir) if ckpt_dir is not None else get_job_run_directory(run_name)
    if ckpt_root.name == "checkpoints":
        dirpath = ckpt_root
    else:
        dirpath = ckpt_root / "checkpoints"
    dirpath.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/MultilabelAveragePrecision_macro",
        dirpath=str(dirpath.resolve()),
        filename=f"{hparams['architecture']}-{hparams['seed']}-{hparams['channels']}-{run_name}-val_mAP_macro-" + "{val/MultilabelAveragePrecision_macro:.2f}",
        save_top_k=1,  # Keep only the best checkpoint (old ones are automatically deleted when a better one is saved)
        save_last=True,  # Always save the last checkpoint as a fallback (saved as 'last.ckpt')
        mode="max",
        auto_insert_metric_name=False,
        enable_version_counter=False,  # remove version counter from filename (v1, v2, ...)
        verbose=True,  # Print when checkpoints are saved/deleted
    )
    
    # Print checkpoint directory and behavior for user reference
    print(f"\n[Checkpoint Configuration]")
    print(f"  Directory: {dirpath.resolve()}")
    print(f"  Best checkpoint pattern: {checkpoint_callback.filename}")
    print(f"  Last checkpoint: last.ckpt (always saved)")
    print(f"  ⚠️  IMPORTANT: With save_top_k=1, old best checkpoints are automatically DELETED")
    print(f"     when a better one is saved. The filename includes the metric value, so it")
    print(f"     changes as the metric improves (e.g., ...-0.76.ckpt → ...-0.77.ckpt)\n")
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val/MultilabelAveragePrecision_macro",
        patience=5,
        verbose=True,
        mode="max",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    
    # Configure accelerator and devices for multi-GPU training
    trainer_kwargs = {
        "max_epochs": 4 if test_run else hparams["epochs"],
        "limit_train_batches": 4 if test_run else None,
        "limit_val_batches": 3 if test_run else None,
        "limit_test_batches": 5 if test_run else None,
        "logger": logger,
        "accelerator": "auto",
        "callbacks": [checkpoint_callback, lr_monitor, early_stopping_callback],
    }
    
    # Add devices if specified
    if devices is not None:
        trainer_kwargs["devices"] = devices
        # Auto-select strategy if not specified and using multiple devices
        if strategy is None and devices > 1:
            # Use DDP for multi-GPU (more efficient than ddp_spawn)
            strategy = "ddp"
    
    # Add strategy if specified
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    
    trainer = pl.Trainer(**trainer_kwargs)
    return trainer


def default_dm(
        hparams,
        data_dirs,
        img_size,
):
    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=hparams["batch_size"],
        num_workers_dataloader=hparams["workers"],
        img_size=(hparams["channels"], img_size, img_size),
    )
    # get the norm_transform to extract mean/std from there
    # we can do this because configilm used default transforms that include normalization with the correct values
    # for the BigEarthNet v2.0 dataset in the BENv2DataModule
    norm_transform = [x for x in dm.train_transform.transforms if isinstance(x, transforms.Normalize)]
    assert len(norm_transform) == 1, "Expected exactly one normalization transform"
    norm_transform = norm_transform[0]
    mean = norm_transform.mean
    std = norm_transform.std
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # torchvision.transforms.RandomRotation(180),
            transforms.Normalize(mean, std),
        ]
    )
    dm.train_transform = train_transforms
    return dm


def get_git_info() -> Dict[str, Optional[str]]:
    """Get git commit hash and branch information."""
    git_info = {
        'commit_hash': None,
        'branch': None,
        'is_dirty': None,
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()
        
        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()
        
        # Check if working directory is dirty
        result = subprocess.run(
            ['git', 'diff', '--quiet'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=5,
        )
        git_info['is_dirty'] = result.returncode != 0
        
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        # Git not available or not a git repository
        pass
    
    return git_info


def get_system_info() -> Dict[str, Any]:
    """Get system information (GPU, CUDA, Python version, etc.)."""
    system_info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': None,
        'gpu_count': 0,
        'gpu_info': [],
    }
    
    if torch.cuda.is_available():
        system_info['cuda_version'] = torch.version.cuda
        system_info['gpu_count'] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            gpu_info = {
                'device_id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
            }
            system_info['gpu_info'].append(gpu_info)
    
    return system_info


def get_environment_info() -> Dict[str, str]:
    """Get installed package versions."""
    env_info = {}
    
    packages_to_check = [
        'torch',
        'lightning',
        'pytorch_lightning',
        'torchvision',
        'numpy',
        'pandas',
        'configilm',
    ]
    
    for package in packages_to_check:
        try:
            if package == 'lightning' or package == 'pytorch_lightning':
                import lightning.pytorch as pl
                env_info[package] = pl.__version__
            elif package == 'torch':
                env_info[package] = torch.__version__
            elif package == 'torchvision':
                import torchvision
                env_info[package] = torchvision.__version__
            elif package == 'numpy':
                import numpy
                env_info[package] = numpy.__version__
            elif package == 'pandas':
                import pandas
                env_info[package] = pandas.__version__
            elif package == 'configilm':
                try:
                    import configilm
                    env_info[package] = getattr(configilm, '__version__', 'unknown')
                except:
                    env_info[package] = 'installed'
        except ImportError:
            env_info[package] = 'not_installed'
    
    return env_info


def get_model_summary(model: Union[pl.LightningModule, torch.nn.Module]) -> Dict[str, Any]:
    """Get model architecture summary."""
    summary = {
        'total_parameters': 0,
        'trainable_parameters': 0,
        'model_size_mb': 0,
    }
    
    try:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary['total_parameters'] = total_params
        summary['trainable_parameters'] = trainable_params
        
        # Estimate model size (rough approximation)
        # Assuming float32 (4 bytes per parameter)
        model_size_bytes = total_params * 4
        summary['model_size_mb'] = model_size_bytes / (1024 ** 2)
        
        # Get model structure info
        if hasattr(model, 'model'):
            # For BigEarthNetv2_0_ImageClassifier
            summary['model_type'] = type(model.model).__name__
        elif hasattr(model, 'dinov3_backbone'):
            # For MultiModalLightningModule
            summary['model_type'] = 'MultiModalModel'
        else:
            summary['model_type'] = type(model).__name__
            
    except Exception as e:
        summary['error'] = str(e)
    
    return summary


def get_data_statistics(data_module: BENv2DataModule) -> Dict[str, Any]:
    """Get dataset statistics."""
    stats = {
        'train_size': None,
        'val_size': None,
        'test_size': None,
        'num_classes': 19,
        'image_size': None,
        'channels': None,
    }
    
    try:
        if hasattr(data_module, 'train_dataset') and data_module.train_dataset is not None:
            stats['train_size'] = len(data_module.train_dataset)
        if hasattr(data_module, 'val_dataset') and data_module.val_dataset is not None:
            stats['val_size'] = len(data_module.val_dataset)
        if hasattr(data_module, 'test_dataset') and data_module.test_dataset is not None:
            stats['test_size'] = len(data_module.test_dataset)
        
        # Get image dimensions from transform or dataset
        if hasattr(data_module, 'img_size'):
            stats['image_size'] = data_module.img_size
        elif hasattr(data_module, 'train_dataset') and hasattr(data_module.train_dataset, 'img_size'):
            stats['image_size'] = data_module.train_dataset.img_size
        
        # Get channels from transform or dataset
        if hasattr(data_module, 'train_transform'):
            # Try to infer from Normalize transform
            norm_transforms = [t for t in data_module.train_transform.transforms if isinstance(t, transforms.Normalize)]
            if norm_transforms:
                stats['channels'] = len(norm_transforms[0].mean)
    except Exception as e:
        stats['error'] = str(e)
    
    return stats


def save_training_metadata(
    run_dir: Path,
    hparams: Dict[str, Any],
    model: Optional[Union[pl.LightningModule, torch.nn.Module]] = None,
    data_module: Optional[BENv2DataModule] = None,
    training_command: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Save comprehensive training metadata to run directory.
    
    Args:
        run_dir: Directory to save metadata
        hparams: Hyperparameters dictionary
        model: Optional model instance for architecture summary
        data_module: Optional data module for dataset statistics
        training_command: Optional training command/arguments
        config_path: Optional path to config file (will be copied if not already copied)
    
    Returns:
        Dictionary mapping metadata type to saved file path
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    timestamp = datetime.now().isoformat()
    
    # 1. Save hyperparameters
    hparams_path = run_dir / "hyperparameters.json"
    with open(hparams_path, 'w') as f:
        json.dump(hparams, f, indent=2, default=str)
    saved_files['hyperparameters'] = hparams_path
    
    # 2. Save git information
    git_info = get_git_info()
    git_path = run_dir / "git_info.json"
    with open(git_path, 'w') as f:
        json.dump(git_info, f, indent=2)
    saved_files['git_info'] = git_path
    
    # 3. Save system information
    system_info = get_system_info()
    system_path = run_dir / "system_info.json"
    with open(system_path, 'w') as f:
        json.dump(system_info, f, indent=2, default=str)
    saved_files['system_info'] = system_path
    
    # 4. Save environment information
    env_info = get_environment_info()
    env_path = run_dir / "environment.json"
    with open(env_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    saved_files['environment'] = env_path
    
    # 5. Save model summary if model provided
    if model is not None:
        model_summary = get_model_summary(model)
        model_path = run_dir / "model_summary.json"
        with open(model_path, 'w') as f:
            json.dump(model_summary, f, indent=2, default=str)
        saved_files['model_summary'] = model_path
    
    # 6. Save data statistics if data module provided
    if data_module is not None:
        data_stats = get_data_statistics(data_module)
        data_path = run_dir / "data_statistics.json"
        with open(data_path, 'w') as f:
            json.dump(data_stats, f, indent=2, default=str)
        saved_files['data_statistics'] = data_path
    
    # 7. Save training command
    if training_command:
        cmd_path = run_dir / "training_command.txt"
        with open(cmd_path, 'w') as f:
            f.write(f"Training started at: {timestamp}\n")
            f.write(f"Command: {training_command}\n")
        saved_files['training_command'] = cmd_path
    
    # 8. Create comprehensive metadata summary
    metadata_summary = {
        'timestamp': timestamp,
        'run_name': hparams.get('run_name', 'unknown'),
        'architecture': hparams.get('architecture', 'unknown'),
        'git': git_info,
        'system': system_info,
        'environment': env_info,
        'hyperparameters': hparams,
    }
    
    if model is not None:
        metadata_summary['model'] = get_model_summary(model)
    
    if data_module is not None:
        metadata_summary['data'] = get_data_statistics(data_module)
    
    summary_path = run_dir / "training_metadata.json"
    with open(summary_path, 'w') as f:
        json.dump(metadata_summary, f, indent=2, default=str)
    saved_files['metadata_summary'] = summary_path
    
    # 9. Save human-readable report
    report_path = run_dir / "training_metadata.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING METADATA REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Run Name: {hparams.get('run_name', 'unknown')}\n")
        f.write(f"Architecture: {hparams.get('architecture', 'unknown')}\n\n")
        
        f.write("="*80 + "\n")
        f.write("GIT INFORMATION\n")
        f.write("="*80 + "\n")
        f.write(f"Commit Hash: {git_info.get('commit_hash', 'N/A')}\n")
        f.write(f"Branch: {git_info.get('branch', 'N/A')}\n")
        f.write(f"Dirty Working Directory: {git_info.get('is_dirty', 'N/A')}\n\n")
        
        f.write("="*80 + "\n")
        f.write("SYSTEM INFORMATION\n")
        f.write("="*80 + "\n")
        f.write(f"Hostname: {system_info.get('hostname', 'N/A')}\n")
        f.write(f"Platform: {system_info.get('platform', 'N/A')}\n")
        f.write(f"Python Version: {system_info.get('python_version', 'N/A')}\n")
        f.write(f"CUDA Available: {system_info.get('cuda_available', False)}\n")
        if system_info.get('cuda_version'):
            f.write(f"CUDA Version: {system_info.get('cuda_version')}\n")
        f.write(f"GPU Count: {system_info.get('gpu_count', 0)}\n")
        for gpu in system_info.get('gpu_info', []):
            f.write(f"  GPU {gpu['device_id']}: {gpu['name']} ({gpu['memory_total_gb']:.2f} GB)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("ENVIRONMENT\n")
        f.write("="*80 + "\n")
        for package, version in env_info.items():
            f.write(f"{package}: {version}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("HYPERPARAMETERS\n")
        f.write("="*80 + "\n")
        for key, value in hparams.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        if model is not None:
            model_summary = get_model_summary(model)
            f.write("="*80 + "\n")
            f.write("MODEL SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Model Type: {model_summary.get('model_type', 'N/A')}\n")
            f.write(f"Total Parameters: {model_summary.get('total_parameters', 0):,}\n")
            f.write(f"Trainable Parameters: {model_summary.get('trainable_parameters', 0):,}\n")
            f.write(f"Model Size (MB): {model_summary.get('model_size_mb', 0):.2f}\n")
            f.write("\n")
        
        if data_module is not None:
            data_stats = get_data_statistics(data_module)
            f.write("="*80 + "\n")
            f.write("DATA STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"Train Size: {data_stats.get('train_size', 'N/A')}\n")
            f.write(f"Val Size: {data_stats.get('val_size', 'N/A')}\n")
            f.write(f"Test Size: {data_stats.get('test_size', 'N/A')}\n")
            f.write(f"Image Size: {data_stats.get('image_size', 'N/A')}\n")
            f.write(f"Channels: {data_stats.get('channels', 'N/A')}\n")
            f.write(f"Number of Classes: {data_stats.get('num_classes', 'N/A')}\n")
            f.write("\n")
        
        if training_command:
            f.write("="*80 + "\n")
            f.write("TRAINING COMMAND\n")
            f.write("="*80 + "\n")
            f.write(f"{training_command}\n")
    
    saved_files['metadata_report'] = report_path
    
    return saved_files
