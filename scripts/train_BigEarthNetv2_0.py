"""
This is an example script for supervised image classification using the BigEarthNet v2.0 dataset.
"""
import sys
from pathlib import Path
from typing import Optional

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
from scripts.utils import upload_model_and_readme_to_hub, get_benv2_dir_dict, get_bands, default_trainer, default_dm

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


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
        resume_from: str = typer.Option(None, help="Path to checkpoint file to resume training from. "
                                                   "Can be a full path or 'best'/'last' to use the best/last checkpoint from the checkpoint directory."),
        head_type: str = typer.Option("linear", help="Classification head type to use for DINOv3 backbones. Options: linear, mlp."),
        head_mlp_dims: str = typer.Option(
            None,
            help="Comma-separated hidden dimensions for the MLP head (e.g., '1024,512'). "
                 "Only used when head_type is 'mlp'.",
        ),
        head_dropout: Optional[float] = typer.Option(
            None,
            help="Dropout probability for the classification head. Defaults to drop_rate when not set.",
        ),
        config_path: str = typer.Option(None, help="Path to config YAML file for data directory configuration. "
                                               "If not provided, will use hostname-based directory selection."),
):
    # DEBUG: Print received parameters
    print(f"\n{'='*80}")
    print(f"DEBUG: train_BigEarthNetv2_0.py - Received parameters:")
    print(f"  architecture: {architecture}")
    print(f"  linear_probe: {linear_probe}")
    print(f"  head_type: {head_type}")
    print(f"  head_mlp_dims: {head_mlp_dims}")
    print(f"  head_dropout: {head_dropout}")
    print(f"{'='*80}\n")
    
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
    }
    trainer = default_trainer(hparams, use_wandb, test_run)

    # Get data directories - from config file if provided, otherwise use hostname
    hostname, data_dirs = get_benv2_dir_dict(config_path=config_path)
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)  # Allow mock data for testing
    dm = default_dm(hparams, data_dirs, img_size)

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
