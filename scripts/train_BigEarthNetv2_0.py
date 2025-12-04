"""
This is an example script for supervised image classification using the BigEarthNet v2.0 dataset.
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
import torch.nn as nn
import typer
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BENv2_utils import resolve_data_dir
from torchvision import transforms

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from scripts.utils import upload_model_and_readme_to_hub, get_benv2_dir_dict, get_bands, default_trainer, default_dm, get_job_run_directory, snapshot_config_file

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


class HybridSatelliteNormalizer(nn.Module):
    """
    Normalization transform for satellite imagery combining 2%-98% robust scaling (with masking) + gamma correction.
    
    Applies:
    1. Per-channel 2%-98% robust scaling (ignores darkest shadows and brightest clouds)
    2. Clipping to [0, 1]
    3. Standard ImageNet normalization (for RGB) or original normalization (for multi-channel)
    """

    def __init__(self, num_channels: int = 3, original_mean=None, original_std=None):
        super().__init__()
        # Standard ImageNet stats (for RGB)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.gamma = 1.0 / 2.2  # linear -> sRGB

        # Original normalization stats (for multi-channel)
        self.num_channels = num_channels
        if original_mean is not None and original_std is not None:
            self.original_mean = torch.tensor(original_mean).view(num_channels, 1, 1)
            self.original_std = torch.tensor(original_std).view(num_channels, 1, 1)
        else:
            self.original_mean = None
            self.original_std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, H, W) tensor or (B, C, H, W) tensor.
               Values are roughly in physical reflectance range.
        """
        added_batch_dim = False
        if x.dim() == 3:
            # (C, H, W) -> (1, C, H, W)
            x = x.unsqueeze(0)
            added_batch_dim = True
        assert x.dim() == 4, "Input to HybridSatelliteNormalizer must be 3D or 4D tensor"

        B, C, H, W = x.shape

        # 1. CREATE MASK: valid pixels (ignore absolute black padding)
        valid_mask = x > 1e-4  # (B, C, H, W)

        # 2. ROBUST SCALING (2%-98%) with masking, per (B, C)
        x_flat = x.view(B, C, -1)
        mask_flat = valid_mask.view(B, C, -1)

        # Defaults if not enough valid pixels
        min_val = torch.zeros(B, C, 1, device=x.device, dtype=x.dtype)
        max_val = torch.ones(B, C, 1, device=x.device, dtype=x.dtype)

        for b in range(B):
            for c in range(C):
                valid_pixels = x_flat[b, c][mask_flat[b, c]]
                n_valid = valid_pixels.numel()
                if n_valid > 100:
                    # 2nd and 98th percentiles on valid pixels only
                    k2 = max(int(0.02 * n_valid), 0) + 1
                    k98 = max(int(0.98 * n_valid), 0) + 1
                    k2 = min(k2, n_valid)
                    k98 = min(k98, n_valid)
                    min_val[b, c] = torch.kthvalue(valid_pixels, k2).values
                    max_val[b, c] = torch.kthvalue(valid_pixels, k98).values

        # Reshape for broadcasting
        min_val = min_val.view(B, C, 1, 1)
        max_val = max_val.view(B, C, 1, 1)

        # 3. Apply scaling
        scale = (max_val - min_val).clamp_min(1e-6)
        x = (x - min_val) / scale

        # 4. Clip to [0, 1]
        x = torch.clamp(x, 0.0, 1.0)

        # 5. Gamma correction (lift mid-tones)
        x = torch.pow(x, self.gamma)

        # 6. Normalization
        if self.num_channels == 3:
            mean = self.imagenet_mean.to(x.device, dtype=x.dtype)
            std = self.imagenet_std.to(x.device, dtype=x.dtype)
        elif self.original_mean is not None and self.original_std is not None:
            mean = self.original_mean.to(x.device, dtype=x.dtype)
            std = self.original_std.to(x.device, dtype=x.dtype)
        else:
            # Per-channel normalization over spatial dims and batch
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
            std = x.std(dim=(0, 2, 3), keepdim=True) + 1e-6

        x = (x - mean) / std

        if added_batch_dim:
            x = x.squeeze(0)
        return x


def main(
        architecture: str = typer.Option("resnet101", help="Model name (timm model name or dinov3-base/dinov3-large/dinov3-small/dinov3-giant)"),
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(512, help="Batch size"),
        drop_rate: float = typer.Option(0.15, help="Dropout rate"),
        drop_path_rate: float = typer.Option(0.15, help="Drop path rate"),
        warmup: int = typer.Option(1000, help="Warmup steps, set to -1 for automatic calculation"),
        workers: int = typer.Option(8, help="Number of workers"),
        bandconfig: str = typer.Option("all",
                                       help="Band configuration, one of all, s2, s1, rgb, all_full, s2_full, s1_full. "
                                            "rgb uses 3-channel RGB (B04, B03, B02) from Sentinel-2."),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        upload_to_hub: bool = typer.Option(False, help="Upload model to Huggingface Hub"),
        test_run: bool = typer.Option(True, help="Run training with fewer epochs and batches"),
        hf_entity: str = typer.Option(None, help="Huggingface entity to upload the model to. Has to be set if "
                                                 "upload_to_hub is True."),
        dinov3_model_name: str = typer.Option(None, help="DINOv3 HuggingFace model name (e.g., facebook/dinov3-base). "
                                                         "If None, will be inferred from architecture parameter."),
        linear_probe: bool = typer.Option(False, help="Freeze DINOv3 backbone and train linear classifier only"),
        head_type: str = typer.Option("linear", help="Classification head type: 'linear' or 'mlp'"),
        head_mlp_dims: str = typer.Option(None, help="Comma-separated MLP hidden dimensions (e.g., '1024,512'). Only used when head_type='mlp'"),
        head_dropout: float = typer.Option(None, help="Dropout rate for classification head. If None, uses drop_rate"),
        resume_from: str = typer.Option(None, help="Path to checkpoint file to resume training from. "
                                                   "Can be a full path or 'best'/'last' to use the best/last checkpoint from the checkpoint directory."),
        config_path: str = typer.Option(None, help="Path to config YAML file for data directory configuration. "
                          "If not provided, will use hostname-based directory selection."),
        run_name: str = typer.Option(None, help="Custom name for this run. Defaults to <architecture>-<bandconfig>-<seed>-<timestamp>"),
        devices: int = typer.Option(None, help="Number of GPUs to use (None = auto-detect)"),
        strategy: str = typer.Option(None, help="Training strategy (None = auto, 'ddp', 'ddp_spawn', etc.)"),
):
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    # FIXED MODEL PARAMETERS
    num_classes = 19
    img_size = 120

    # HUGGINGFACE MODEL PARAMETERS
    version = "v0.2.0"
    if upload_to_hub and hf_entity is None:
        raise ValueError("Please specify a Huggingface entity to upload the model to.")

    # set seed
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    if upload_to_hub:
        assert Path("~/.cache/huggingface/token").expanduser().exists(), "Please login to Huggingface Hub first."

    bands, channels = get_bands(bandconfig)
    # fixed model parameters based on the BigEarthNet v2.0 dataset
    ilm_config = ILMConfiguration(
        network_type=ILMType.IMAGE_CLASSIFICATION,
        classes=num_classes,
        image_size=img_size,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        timm_model_name=architecture,
        channels=channels,
    )
    warmup = None if warmup == -1 else warmup
    assert warmup is None or warmup > 0, "Warmup steps must be positive or -1 for automatic calculation"

    # Determine DINOv3 model name if needed
    dinov3_name = dinov3_model_name
    if architecture.startswith('dinov3') and dinov3_name is None:
        # Map architecture names to HuggingFace model names
        # DINOv3 uses specific naming: facebook/dinov3-vit{s|b|l|g}16-pretrain-lvd1689m
        if 'small' in architecture.lower() or 's' in architecture.lower():
            dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        elif 'base' in architecture.lower() or 'b' in architecture.lower():
            dinov3_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        elif 'large' in architecture.lower() or 'l' in architecture.lower():
            dinov3_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        elif 'giant' in architecture.lower() or 'g' in architecture.lower():
            dinov3_name = "facebook/dinov3-vitg16-pretrain-lvd1689m"
        else:
            dinov3_name = "facebook/dinov3-vits16-pretrain-lvd1689m"  # default to small

    # Use the ILMConfiguration object created above (ilm_config)
    mlp_dims = None
    if head_mlp_dims:
        mlp_dims = [int(dim.strip()) for dim in head_mlp_dims.split(",") if dim.strip()]
    head_dropout_val = head_dropout if head_dropout is not None else drop_rate

    model = BigEarthNetv2_0_ImageClassifier(
        ilm_config,
        lr=lr,
        warmup=warmup,
        dinov3_model_name=dinov3_name,
        linear_probe=linear_probe,
        head_type=head_type,
        mlp_hidden_dims=mlp_dims,
        head_dropout=head_dropout_val,
    )

    # Generate unique run name if not provided
    if run_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{architecture}_{bandconfig}_{seed}_{timestamp}"

    # Create run directory and snapshot config YAML for reproducibility
    run_dir = get_job_run_directory(run_name)
    copied_config = snapshot_config_file(config_path, run_dir)
    if copied_config:
        print(f"[Config Snapshot] Copied {copied_config} into run directory for reproducibility.")

    hparams = {
        "architecture": architecture,
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "batch_size": bs,
        "workers": workers,
        "channels": channels,
        "dropout": drop_rate,
        "drop_path_rate": drop_path_rate,
        "bandconfig": bandconfig,
        "warmup": warmup,
        "version": version,
        "linear_probe": linear_probe,
        "head_type": head_type,
        "head_mlp_dims": mlp_dims,
        "head_dropout": head_dropout_val,
        "run_name": run_name,
    }
    trainer = default_trainer(hparams, use_wandb, test_run, devices=devices, strategy=strategy)

    # Get data directories - from config file if provided, otherwise use hostname
    hostname, data_dirs = get_benv2_dir_dict(config_path=config_path)
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)  # Allow mock data for testing
    dm = default_dm(hparams, data_dirs, img_size)
    
    # Apply masked robust scaling + gamma normalization for satellite imagery
    # Extract original normalization stats before replacing
    original_norm_transform = None
    for transform in dm.train_transform.transforms:
        if isinstance(transform, transforms.Normalize):
            original_norm_transform = transform
            break
    
    # Create HybridSatelliteNormalizer with appropriate channel count and original stats
    if original_norm_transform is not None:
        original_mean = original_norm_transform.mean.tolist() if hasattr(original_norm_transform.mean, 'tolist') else list(original_norm_transform.mean)
        original_std = original_norm_transform.std.tolist() if hasattr(original_norm_transform.std, 'tolist') else list(original_norm_transform.std)
        satellite_normalizer = HybridSatelliteNormalizer(
            num_channels=channels,
            original_mean=original_mean,
            original_std=original_std
        )
    else:
        satellite_normalizer = HybridSatelliteNormalizer(num_channels=channels)
    
    # Set to eval mode (no gradients needed for transforms)
    satellite_normalizer.eval()
    
    # Update train transform: keep augmentation, replace Normalize with SatelliteNormalizer
    train_transforms_list = []
    for transform in dm.train_transform.transforms:
        if isinstance(transform, transforms.Normalize):
            # Replace Normalize with SatelliteNormalizer
            train_transforms_list.append(satellite_normalizer)
        else:
            # Keep other transforms (augmentations)
            train_transforms_list.append(transform)
    dm.train_transform = transforms.Compose(train_transforms_list)
    
    # Also update validation and test transforms for consistency
    if hasattr(dm, 'val_transform') and dm.val_transform is not None:
        val_transforms_list = []
        for transform in dm.val_transform.transforms:
            if isinstance(transform, transforms.Normalize):
                val_transforms_list.append(satellite_normalizer)
            else:
                val_transforms_list.append(transform)
        dm.val_transform = transforms.Compose(val_transforms_list)
    
    if hasattr(dm, 'test_transform') and dm.test_transform is not None:
        test_transforms_list = []
        for transform in dm.test_transform.transforms:
            if isinstance(transform, transforms.Normalize):
                test_transforms_list.append(satellite_normalizer)
            else:
                test_transforms_list.append(transform)
        dm.test_transform = transforms.Compose(test_transforms_list)
    
    print(f"Applied masked 2%-98% scaling + gamma (HybridSatelliteNormalizer) to dataset transforms (channels={channels})")

    # Handle checkpoint resume
    ckpt_path = None
    if resume_from is not None:
        if resume_from.lower() in ["best", "last"]:
            # Use Lightning's built-in checkpoint resolution
            ckpt_path = resume_from.lower()
            print(f"Resuming from {resume_from} checkpoint (will be resolved by Lightning)")
        else:
            # Use provided checkpoint path
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

    trainer.fit(model, dm, ckpt_path=ckpt_path)
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    model_name = f"{architecture}-{bandconfig}-{version}"
    model.save_pretrained(f"hf_models/{model_name}", config=ilm_config)
    # Use run_name to prevent conflicts when running multiple trainings
    model_name = f"{architecture}-{bandconfig}-{run_name}-{version}"
    model.save_pretrained(f"hf_models/{model_name}", config=ilm_config)

    print("=== Training finished ===")
    # upload_model_and_readme_to_hub(
    #     model=model,
    #     model_name=model_name,
    #     hf_entity=hf_entity,
    #     test_results=results[0],
    #     hparams=hparams,
    #     trainer=trainer,
    #     upload=upload_to_hub,
    # )


if __name__ == "__main__":
    typer.run(main)
