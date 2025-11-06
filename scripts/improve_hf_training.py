import sys
from warnings import warn
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
from configilm.extra.BENv2_utils import resolve_data_dir

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from scripts.load_BigEarthNetv2_0_pretrained_from_hf import download_and_evaluate_model
from scripts.utils import upload_model_and_readme_to_hub, get_benv2_dir_dict, get_bands, default_trainer, default_dm


def get_arch_version_bandconfig(model_name: str, config: ILMConfiguration):
    architecture = model_name.split("/")[-1].split("-")[-3]
    assert architecture == config.timm_model_name, f"Model name {architecture} does not match config {config.timm_model_name}"
    version = model_name.split("/")[-1].split("-")[-1]
    bandconfig = model_name.split("/")[-1].split("-")[-2]
    if bandconfig == "s2":
        assert config.channels == 10, f"Bandconfig {bandconfig} does not match config {config.channels}"
    elif bandconfig == "s1":
        assert config.channels == 2, f"Bandconfig {bandconfig} does not match config {config.channels}"
    elif bandconfig == "all":
        assert config.channels == 12, f"Bandconfig {bandconfig} does not match config {config.channels}"
    elif bandconfig == "rgb":
        assert config.channels == 3, f"Bandconfig {bandconfig} does not match config {config.channels}"
    else:
        raise ValueError(f"Unknown band configuration {bandconfig}")
    return architecture, version, bandconfig


def train_new_model(
        config: ILMConfiguration,
        comparison_model_name: str,
        lr: float,
        epochs: int,
        bs: int,
        drop_rate: float,
        drop_path_rate: float,
        warmup: int,
        workers: int,
        use_wandb: bool,
        test_run: bool,
        seed: int,
):
    architecture, version, bandconfig = get_arch_version_bandconfig(comparison_model_name, config)

    bands, channels = get_bands(bandconfig)
    assert channels == config.channels, f"Bandconfig {bandconfig} does not match config {config.channels}"

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
    dm = default_dm(hparams, data_dirs, config.image_size)

    print(f"Using data directories for {hostname}")

    trainer.fit(model, dm)
    return model, dm, trainer, hparams


def main(
        model_name: str = typer.Option("hackelle/resnet18-s2-v0.1.1", help="Model name"),
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(32, help="Batch size"),
        drop_rate: float = typer.Option(0.15, help="Dropout rate"),
        drop_path_rate: float = typer.Option(0.15, help="Drop path rate"),
        warmup: int = typer.Option(1000, help="Warmup steps, set to -1 for automatic calculation"),
        workers: int = typer.Option(8, help="Number of workers"),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        test_run: bool = typer.Option(True, help="Run training and eval with fewer epochs and batches"),
        hf_entity: str = typer.Option(None, help="Huggingface entity to upload the model to. Has to be set if "
                                                 "the final upload should work"),
):
    assert Path("~/.cache/huggingface/token").expanduser().exists(), "Please login to Huggingface Hub first."
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    if hf_entity is None:
        warn("No Huggingface entity specified. The model will not be uploaded to Huggingface Hub.")
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # train the model with the given hyperparameters based on the config of the downloaded model
    config = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_name).config

    architecture, version, bandconfig = get_arch_version_bandconfig(model_name, config)
    new_model_name = f"{architecture}-{bandconfig}-{version}"
    if new_model_name != model_name.split("/")[-1]:
        warn(f"New model name {new_model_name} does not match the old model name {model_name.split('/')[-1]}")

    model, dm, trainer, hparams = train_new_model(
        config=config,
        comparison_model_name=model_name,
        lr=lr,
        epochs=epochs,
        bs=bs,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        warmup=warmup,
        workers=workers,
        use_wandb=use_wandb,
        test_run=test_run,
        seed=seed,
    )

    new_results = trainer.test(model, datamodule=dm, ckpt_path="best")
    print("=== Training finished ===")

    # comparison and upload
    new_metric = new_results[0]["test/MultilabelAveragePrecision_macro"]

    # load the model from Huggingface Hub and evaluate to get a baseline
    compare_results = download_and_evaluate_model(
        model_name=model_name,
        limit_test_batches=5 if test_run else None,
        batch_size=bs,
        num_workers_dataloader=workers,
    )
    compare_metric = compare_results["AveragePrecision"]["macro"]

    print(f"=== Results ===")
    print(f"Compare metric: {compare_metric:.4f}")
    print(f"New metric: {new_metric:.4f}")
    if new_metric > compare_metric:
        print(f"New model improved the compare metric by {new_metric - compare_metric:.4f}")
        upload_model_and_readme_to_hub(
            model=model,
            model_name=new_model_name,
            hf_entity=hf_entity,
            test_results=new_results[0],
            hparams=hparams,
            trainer=trainer,
            upload=True,
        )
    else:
        print("=== Skipping upload to Huggingface Hub because the new model did not improve the compare metric ===")


if __name__ == "__main__":
    typer.run(main)
