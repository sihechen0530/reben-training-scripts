"""
This is an example script for supervised image classification using the BigEarthNet v2.0 dataset.
"""

from pathlib import Path

import lightning.pytorch as pl
import torch
import typer
from configilm.ConfigILM import ILMConfiguration
from configilm.ConfigILM import ILMType
from configilm.extra.BENv2_utils import STANDARD_BANDS, resolve_data_dir
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from lightning.pytorch.loggers import WandbLogger

from ben_publication.BENv2ImageClassifier import BENv2ImageEncoder

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"

BASE_DIR = Path("~/data").expanduser()
BENv2_DIR = BASE_DIR / "BigEarthNet-V2"

BENv2_DIR_DICT = {
    "images_lmdb": BENv2_DIR / "BigEarthNet-V2-LMDB",
    "split_csv": BENv2_DIR / "patch_id_split_mapping.csv",
    "s1_mapping_csv": BENv2_DIR / "patch_id_s1_mapping.csv",
    "labels_csv": BENv2_DIR / "patch_id_label_mapping.csv",
}


def main(
        architecture: str = typer.Option("resnet18", help="Model name"),
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(16, help="Batch size"),
        workers: int = typer.Option(8, help="Number of workers"),
        bandconfig: str = typer.Option("all",
                                       help="Band configuration, one of all, s2, s1, all_full, s2_full, s1_full"),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        upload_to_hub: bool = typer.Option(True, help="Upload model to Huggingface Hub"),
):
    # set seed
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")

    if upload_to_hub:
        assert Path("~/.cache/huggingface/token").expanduser().exists(), "Please login to Huggingface Hub first."

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
    else:
        raise ValueError(
            f"Unknown band configuration {bandconfig}, select one of all, s2, s1 or all_full, s2_full, s1_full. The "
            f"full versions include all bands whereas the non-full versions only include the 10m & 20m bands."
        )
    channels = len(bands)
    data_dirs = resolve_data_dir(BENv2_DIR_DICT, allow_mock=True)

    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=bs,
        num_workers_dataloader=workers,
        img_size=(channels, 120, 120),
    )

    # fixed model parameters based on the BigEarthNet v2.0 dataset
    num_classes = 19
    img_size = 120
    dropout = 0.25
    config = ILMConfiguration(
        network_type=ILMType.IMAGE_CLASSIFICATION,
        classes=num_classes,
        image_size=img_size,
        drop_rate=dropout,
        timm_model_name=architecture,
        channels=channels,
    )

    model = BENv2ImageEncoder(config, lr=lr)

    # we assume, that we are already logged in to wandb
    if use_wandb:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=True)
    else:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=False, mode="disabled")
    logger.log_hyperparams(
        {
            "architecture": architecture,
            "seed": seed,
            "lr": lr,
            "epochs": epochs,
            "batch_size": bs,
            "workers": workers,
            "channels": channels,
        }
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/MultilabelAveragePrecision_macro",
        dirpath="./checkpoints",
        filename=f"{architecture}-{seed}-{channels}-val_mAP_macro-" + "{val/MultilabelAveragePrecision_macro:.2f}",
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
        max_epochs=2 if not use_wandb else epochs,
        limit_train_batches=4 if not use_wandb else None,
        limit_val_batches=3 if not use_wandb else None,
        limit_test_batches=5 if not use_wandb else None,
        logger=logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    print("=== Training finished ===")
    if upload_to_hub:
        print("=== Uploading model to Huggingface Hub ===")
        version = "v0.1.1-alpha"
        model_name = f"BENv2-{architecture}-{seed}-{bandconfig}-{version}"
        print(f"Uploading model as {model_name}")
        model.save_pretrained(f"hf_models/{model_name}", config=config)
        model.push_to_hub(model_name, commit_message=f"Upload {model_name}")
        print("=== Done ===")
    else:
        print("=== Skipping upload to Huggingface Hub ===")


if __name__ == "__main__":
    typer.run(main)
