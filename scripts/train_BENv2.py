"""
This is an example script for supervised image classification using the BigEarthNet v2.0 dataset.
"""
import socket
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
from ben_publication.mock_dm import MockDataModule

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"

BENv2_DIR_MARS = Path("/data/kaiclasen")
BENv2_DIR_DICT_MARS = {
    "images_lmdb": BENv2_DIR_MARS / "BENv2.lmdb",
    "split_csv": BENv2_DIR_MARS / "patch_id_split_mapping.csv",
    "s1_mapping_csv": BENv2_DIR_MARS / "patch_id_s1_mapping.csv",
    "labels_csv": BENv2_DIR_MARS / "patch_id_label_mapping.csv",
}

BENv2_DIR_ERDE = Path("/faststorage/BigEarthNet-V2")
BENv2_DIR_DICT_ERDE = {
    "images_lmdb": BENv2_DIR_ERDE / "BigEarthNet-V2-LMDB",
    "split_csv": BENv2_DIR_ERDE / "patch_id_split_mapping.csv",
    "s1_mapping_csv": BENv2_DIR_ERDE / "patch_id_s1_mapping.csv",
    "labels_csv": BENv2_DIR_ERDE / "patch_id_label_mapping.csv",
}

BENv2_DIR_PLUTO = Path("/home/kaiclasen/bigearthnet-pipeline")
BENv2_DIR_DICT_PLUTO = {
    "images_lmdb": BENv2_DIR_PLUTO / "artifacts-lmdb" / "BigEarthNet-V2",
    "split_csv": BENv2_DIR_PLUTO / "artifacts-result" / "patch_id_split_mapping.csv",
    "s1_mapping_csv": BENv2_DIR_PLUTO / "artifacts-result" / "patch_id_s1_mapping.csv",
    "labels_csv": BENv2_DIR_PLUTO / "artifacts-result" / "patch_id_label_mapping.csv",
}

BENv2_DIR_DEFAULT = Path("~/data/BigEarthNet-V2").expanduser()
BENv2_DIR_DICT_DEFAULT = {
    "images_lmdb": BENv2_DIR_DEFAULT / "BigEarthNet-V2-LMDB",
    "split_csv": BENv2_DIR_DEFAULT / "patch_id_split_mapping.csv",
    "s1_mapping_csv": BENv2_DIR_DEFAULT / "patch_id_s1_mapping.csv",
    "labels_csv": BENv2_DIR_DEFAULT / "patch_id_label_mapping.csv",
}

BENv2_DIR_DICTS = {
    'mars.rsim.tu-berlin.de': BENv2_DIR_DICT_PLUTO,
    'erde': BENv2_DIR_DICT_ERDE,
    'pluto': BENv2_DIR_DICT_PLUTO,
    'default': BENv2_DIR_DICT_DEFAULT,
}


def _get_benv2_dir_dict() -> tuple[str, dict]:
    hostname = socket.gethostname()
    return hostname, BENv2_DIR_DICTS.get(hostname, BENv2_DIR_DICT_DEFAULT)


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
        upload_to_hub: bool = typer.Option(False, help="Upload model to Huggingface Hub"),
        test_run: bool = typer.Option(True, help="Run training with fewer epochs and batches"),
):
    # FIXED MODEL PARAMETERS
    num_classes = 19
    img_size = 120
    dropout = 0.375

    # HUGGINGFACE MODEL PARAMETERS
    version = "v0.1.1"
    hf_entity = "BIFOLD-BigEarthNetv2-0"  # e.g. your username or organisation
    # you can set it to None if it should be uploaded to the logged in user

    # set seed
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("medium")
    torch.cuda.set_per_process_memory_fraction(2/80)

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
    hostname, data_dirs = _get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
    print(f"Using data directories for {hostname}")

    use_mock_data = True
    if use_mock_data:
        dm = MockDataModule(
            dims=(channels, img_size, img_size),
            clss=num_classes,
            train_length=200_000,
            val_length=120_000,
            test_length=120_000,
            bs=bs,
            num_workers=workers,
        )
    else:
        dm = BENv2DataModule(
            data_dirs=data_dirs,
            batch_size=bs,
            num_workers_dataloader=workers,
            img_size=(channels, img_size, img_size),
        )

    # fixed model parameters based on the BigEarthNet v2.0 dataset
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
            "dropout": dropout,
            "version": version,
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
        max_epochs=4 if test_run else epochs,
        limit_train_batches=4 if test_run else None,
        limit_val_batches=3 if test_run else None,
        limit_test_batches=5 if test_run else None,
        logger=logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    print("=== Training finished ===")
    if upload_to_hub:
        print("=== Uploading model to Huggingface Hub ===")
        model_name = f"BENv2-{architecture}-{seed}-{bandconfig}-{version}"
        print(f"Uploading model as {model_name}")
        model.save_pretrained(f"hf_models/{model_name}", config=config)
        push_path = f"{hf_entity}/{model_name}" if hf_entity else model_name
        print(f"Pushing to {push_path}")
        model.push_to_hub(push_path, commit_message=f"Upload {model_name}")
        print("=== Done ===")
    else:
        print("=== Skipping upload to Huggingface Hub ===")


if __name__ == "__main__":
    typer.run(main)
