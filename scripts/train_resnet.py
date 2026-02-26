"""
Training script for ResNet with customizable number of channels.
Supports easy modification of hyperparameters via commandline arguments.
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
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BENv2_utils import resolve_data_dir

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from scripts.utils import upload_model_and_readme_to_hub, get_benv2_dir_dict, get_bands, default_trainer, default_dm

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def main(
        architecture: str = typer.Option("resnet18", help="ResNet architecture (resnet18, resnet34, resnet50, etc.)"),
        num_channels: int = typer.Option(None, help="Number of input channels. If None, uses bandconfig to determine."),
        bandconfig: str = typer.Option("s2", help="Band configuration (all, s2, s1, rgb, etc.). "
                                                   "Ignored if num_channels is specified."),
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(32, help="Batch size"),
        drop_rate: float = typer.Option(0.15, help="Dropout rate"),
        drop_path_rate: float = typer.Option(0.0, help="Drop path rate"),
        warmup: int = typer.Option(1000, help="Warmup steps, set to -1 for automatic calculation"),
        workers: int = typer.Option(8, help="Number of workers"),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        upload_to_hub: bool = typer.Option(False, help="Upload model to Huggingface Hub"),
        test_run: bool = typer.Option(True, help="Run training with fewer epochs and batches"),
        hf_entity: str = typer.Option(None, help="Huggingface entity to upload the model to. Has to be set if "
                                                 "upload_to_hub is True."),
        resume_from: str = typer.Option(None, help="Path to checkpoint file to resume training from. "
                                                   "Can be a full path or 'best'/'last' to use the best/last checkpoint."),
):
    """
    Train a ResNet model on BigEarthNet v2.0 with customizable number of channels.
    
    Examples:
        # Train ResNet18 on 10 channels (S2 bands)
        python train_resnet.py --architecture resnet18 --num_channels 10 --epochs 50 --lr 0.001
        
        # Train ResNet50 on 3 channels (RGB)
        python train_resnet.py --architecture resnet50 --bandconfig rgb --epochs 100 --bs 64
        
        # Train with custom hyperparameters
        python train_resnet.py --architecture resnet34 --num_channels 12 --epochs 200 --lr 0.0005 --bs 128
    """
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

    # Determine number of channels
    if num_channels is not None:
        channels = num_channels
        bands = None  # Will use first num_channels bands
        print(f"Using {channels} channels (specified directly)")
    else:
        bands, channels = get_bands(bandconfig)
        print(f"Using {channels} channels from bandconfig '{bandconfig}'")
    
    # Create model configuration
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
    assert warmup is None or warmup > 0, "Warmup steps must be positive or -1 for automatic calculation"

    model = BigEarthNetv2_0_ImageClassifier(config, lr=lr, warmup=warmup)

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
        "bandconfig": bandconfig if num_channels is None else f"custom_{num_channels}",
        "warmup": warmup,
        "version": version,
    }
    trainer = default_trainer(hparams, use_wandb, test_run)

    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=False)
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
    model_name = f"{architecture}-{hparams['bandconfig']}-{version}"
    model.save_pretrained(f"hf_models/{model_name}", config=config)

    print("=== Training finished ===")
    upload_model_and_readme_to_hub(
        model=model,
        model_name=model_name,
        hf_entity=hf_entity,
        test_results=results[0],
        hparams=hparams,
        trainer=trainer,
        upload=upload_to_hub,
    )


if __name__ == "__main__":
    typer.run(main)

