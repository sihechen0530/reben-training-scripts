import lightning.pytorch as pl
import torch
import typer
from configilm.ConfigILM import ILMConfiguration
from configilm.extra.BENv2_utils import resolve_data_dir
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms

from ben_publication.BENv2ImageClassifier import BENv2ImageEncoder
from scripts.load_BENv2_pretrained_from_hf import download_and_evaluate_model
from scripts.train_BENv2 import _get_benv2_dir_dict


def get_arch_version_bandconfig(model_name: str, config: ILMConfiguration):
    architecture = model_name.split("/")[-1].split("-")[1]
    assert architecture == config.timm_model_name, f"Model name {architecture} does not match config {config.timm_model_name}"
    version = model_name.split("/")[-1].split("-")[-1]
    bandconfig = model_name.split("/")[-1].split("-")[3]
    if bandconfig == "s2":
        assert config.channels == 10, f"Bandconfig {bandconfig} does not match config {config.channels}"
    elif bandconfig == "s1":
        assert config.channels == 2, f"Bandconfig {bandconfig} does not match config {config.channels}"
    elif bandconfig == "all":
        assert config.channels == 12, f"Bandconfig {bandconfig} does not match config {config.channels}"
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

    model = BENv2ImageEncoder(config, lr=lr, warmup=warmup)

    # we assume, that we are already logged in to wandb
    if use_wandb:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=True)
    else:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=False, mode="disabled")
    logger.log_hyperparams(
        {
            "architecture": config.timm_model_name,
            "seed": seed,
            "lr": lr,
            "epochs": epochs,
            "batch_size": bs,
            "workers": workers,
            "channels": config.channels,
            "dropout": drop_rate,
            "drop_path_rate": drop_path_rate,
            "bandconfig": bandconfig,
            "warmup": warmup,
            "version": version,
        }
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/MultilabelAveragePrecision_macro",
        dirpath="./checkpoints",
        filename=f"{architecture}-{seed}-{config.channels}-val_mAP_macro-" + "{val/MultilabelAveragePrecision_macro:.2f}",
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
        enable_version_counter=False,  # remove version counter from filename (v1, v2, ...)
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val/MultilabelAveragePrecision_macro",
        patience=5,
        verbose=True,
        mode="max",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=4 if test_run else epochs,
        limit_train_batches=4 if test_run else None,
        limit_val_batches=3 if test_run else None,
        limit_test_batches=5 if test_run else None,
        logger=logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
    )

    hostname, data_dirs = _get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
    print(f"Using data directories for {hostname}")

    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=bs,
        num_workers_dataloader=workers,
        img_size=(config.channels, config.image_size, config.image_size),
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
            transforms.Normalize(mean, std),
        ]
    )
    dm.train_transform = train_transforms

    trainer.fit(model, dm)
    return model, dm, trainer


def main(
        model_name: str = typer.Option("BIFOLD-BigEarthNetv2-0/BENv2-resnet50-42-s2-v0.1.1", help="Model name"),
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(32, help="Batch size"),
        drop_rate: float = typer.Option(0.375, help="Dropout rate"),
        drop_path_rate: float = typer.Option(0.0, help="Drop path rate"),
        warmup: int = typer.Option(-1, help="Warmup steps, set to -1 for automatic calculation"),
        workers: int = typer.Option(8, help="Number of workers"),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        test_run: bool = typer.Option(True, help="Run training and eval with fewer epochs and batches")
):
    upload_hf_entity = "BIFOLD-BigEarthNetv2-0"
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")

    # load the model from Huggingface Hub and evaluate to get a baseline
    compare_results = download_and_evaluate_model(
        model_name=model_name,
        limit_test_batches=5 if test_run else None,
        batch_size=bs,
        num_workers_dataloader=workers,
    )
    compare_metric = compare_results["AveragePrecision"]["macro"]

    # train the model with the given hyperparameters based on the config of the downloaded model
    config = BENv2ImageEncoder.from_pretrained(model_name).config

    model, dm, trainer = train_new_model(
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
    new_metric = new_results[0]["test/MultilabelAveragePrecision_macro"]

    print(f"=== Results ===")
    print(f"Compare metric: {compare_metric:.4f}")
    print(f"New metric: {new_metric:.4f}")
    if new_metric > compare_metric:
        print(f"New model improved the compare metric by {new_metric - compare_metric:.4f}")
        print("=== Uploading model to Huggingface Hub ===")
        architecture, version, bandconfig = get_arch_version_bandconfig(model_name, config)
        new_model_name = f"BENv2-{architecture}-{seed}-{bandconfig}-{version}"
        assert model_name == new_model_name, f"Model name {model_name} does not match new model name {new_model_name}"
        if upload_hf_entity:
            print(f"Uploading model as {model_name}")
            model.save_pretrained(f"hf_models/{model_name}", config=config)
            push_path = f"{upload_hf_entity}/{model_name}" if upload_hf_entity else model_name
            print(f"Pushing to {push_path}")
            model.push_to_hub(push_path, commit_message=f"Upload {model_name}")
            print("=== Done ===")
        else:
            print("=== Skipping upload to Huggingface Hub because no entity was provided ===")
    else:
        print("=== Skipping upload to Huggingface Hub because the new model did not improve the compare metric ===")

    print("=== Training finished ===")


if __name__ == "__main__":
    typer.run(main)
