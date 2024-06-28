"""
This is an example script for supervised image classification using the BigEarthNet v2.0 dataset.
"""
from pathlib import Path

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
        architecture: str = typer.Option("resnet18", help="Model name"),
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(32, help="Batch size"),
        drop_rate: float = typer.Option(0.15, help="Dropout rate"),
        drop_path_rate: float = typer.Option(0.15, help="Drop path rate"),
        warmup: int = typer.Option(1000, help="Warmup steps, set to -1 for automatic calculation"),
        workers: int = typer.Option(8, help="Number of workers"),
        bandconfig: str = typer.Option("all",
                                       help="Band configuration, one of all, s2, s1, all_full, s2_full, s1_full"),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        upload_to_hub: bool = typer.Option(False, help="Upload model to Huggingface Hub"),
        test_run: bool = typer.Option(True, help="Run training with fewer epochs and batches"),
        hf_entity: str = typer.Option(None, help="Huggingface entity to upload the model to. Has to be set if "
                                                 "upload_to_hub is True."),
):
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    # FIXED MODEL PARAMETERS
    num_classes = 19
    img_size = 120

    # HUGGINGFACE MODEL PARAMETERS
    version = "v0.1.1"
    if upload_to_hub and hf_entity is None:
        raise ValueError("Please specify a Huggingface entity to upload the model to.")

    # set seed
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    if upload_to_hub:
        assert Path("~/.cache/huggingface/token").expanduser().exists(), "Please login to Huggingface Hub first."

    bands, channels = get_bands(bandconfig)
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
        "bandconfig": bandconfig,
        "warmup": warmup,
        "version": version,
    }
    trainer = default_trainer(hparams, use_wandb, test_run)

    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
    dm = default_dm(hparams, data_dirs, img_size)

    trainer.fit(model, dm)
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    model_name = f"{architecture}-{bandconfig}-{version}"
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
