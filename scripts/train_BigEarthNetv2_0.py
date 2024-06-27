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
from huggingface_hub import HfApi
from lightning.pytorch.loggers import WandbLogger
from torchvision import transforms

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"

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

BENv2_DIR_PLUTO = Path("/home/kaiclasen/bigearthnet-pipeline/artifacts")
BENv2_DIR_DICT_PLUTO = {
    "images_lmdb": BENv2_DIR_PLUTO / "artifacts-lmdb" / "BigEarthNet-V2",
    "metadata_parquet": BENv2_DIR_PLUTO / "artifacts-result" / "metadata.parquet",
    "metadata_snow_cloud_parquet": BENv2_DIR_PLUTO / "artifacts-result" / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

BENv2_DIR_DEFAULT = Path("~/data/BigEarthNet-V2").expanduser()
BENv2_DIR_DICT_DEFAULT = {
    "images_lmdb": BENv2_DIR_DEFAULT / "BigEarthNet-V2-LMDB",
    "metadata_parquet": BENv2_DIR_DEFAULT / "metadata.parquet",
    "metadata_snow_cloud_parquet": BENv2_DIR_DEFAULT / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}

BENv2_DIR_DICTS = {
    'mars.rsim.tu-berlin.de': BENv2_DIR_DICT_MARS,
    'erde': BENv2_DIR_DICT_ERDE,
    'pluto': BENv2_DIR_DICT_PLUTO,
    'default': BENv2_DIR_DICT_DEFAULT,
}


def _get_benv2_dir_dict() -> tuple[str, dict]:
    hostname = socket.gethostname()
    return hostname, BENv2_DIR_DICTS.get(hostname, BENv2_DIR_DICT_DEFAULT)


def _infere_example(S1_file: str, S2_file: str, bands: list[str], model: BigEarthNetv2_0_ImageClassifier):
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
        img = torch.cat([img[:,0], img[:,1], img[:, 0, :, :] / img[:, 1, :, :]]).unsqueeze(0)
    elif img.shape[1] == 10:
        # S2 image, use only first 3 bands
        img = img[:,:3]
    elif img.shape[1] == 12:
        # S2 + S1 image, use only first 3 bands of S2
        img = img[:,:3]

    input_rgb = img.squeeze().numpy()[[2, 1, 0], :, :]
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
        vis_bands = "only RGB bands from Sentinel-2"
    elif bandconfig == "s2":
        bands = STANDARD_BANDS[10]  # 10m + 20m Sentinel-2
        vis_bands = "only RGB bands from Sentinel-2"
    elif bandconfig == "s1":
        bands = STANDARD_BANDS[2]  # Sentinel-1
        vis_bands = "VV, VH and VV/VH bands from Sentinel-1"
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


def main(
        architecture: str = typer.Option("resnet18", help="Model name"),
        seed: int = typer.Option(42, help="Random seed"),
        lr: float = typer.Option(0.001, help="Learning rate"),
        epochs: int = typer.Option(100, help="Number of epochs"),
        bs: int = typer.Option(32, help="Batch size"),
        drop_rate: float = typer.Option(0.375, help="Dropout rate"),
        drop_path_rate: float = typer.Option(0.0, help="Drop path rate"),
        warmup: int = typer.Option(-1, help="Warmup steps, set to -1 for automatic calculation"),
        workers: int = typer.Option(8, help="Number of workers"),
        bandconfig: str = typer.Option("all",
                                       help="Band configuration, one of all, s2, s1, all_full, s2_full, s1_full"),
        use_wandb: bool = typer.Option(False, help="Use wandb for logging"),
        upload_to_hub: bool = typer.Option(False, help="Upload model to Huggingface Hub"),
        test_run: bool = typer.Option(True, help="Run training with fewer epochs and batches"),
):
    assert Path(".").resolve().name == "scripts", \
        "Please run this script from the scripts directory. Otherwise some relative paths might not work."
    # FIXED MODEL PARAMETERS
    num_classes = 19
    img_size = 120

    # HUGGINGFACE MODEL PARAMETERS
    version = "v0.1.1"
    hf_entity = "BIFOLD-BigEarthNetv2-0"  # e.g. your username or organisation
    # you can set it to None if it should be uploaded to the logged-in user

    # set seed
    pl.seed_everything(seed, workers=True)
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
    hostname, data_dirs = _get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
    print(f"Using data directories for {hostname}")

    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=bs,
        num_workers_dataloader=workers,
        img_size=(channels, img_size, img_size),
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

    # we assume, that we are already logged in to wandb
    if use_wandb:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=True)
    else:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=False, mode="disabled")
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
    logger.log_hyperparams(hparams)

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
        max_epochs=2 if test_run else epochs,
        limit_train_batches=3 if test_run else None,
        limit_val_batches=2 if test_run else None,
        limit_test_batches=4 if test_run else None,
        logger=logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
    )

    trainer.fit(model, dm)
    results = trainer.test(model, datamodule=dm, ckpt_path="best")
    model_name = f"{architecture}-{bandconfig}-{version}"
    model.save_pretrained(f"hf_models/{model_name}", config=config)

    print("=== Training finished ===")
    if upload_to_hub:
        print("=== Uploading model to Huggingface Hub ===")
        print(f"Uploading model as {model_name}")
        push_path = f"{hf_entity}/{model_name}" if hf_entity is not None else model_name
        print(f"Pushing to {push_path}")
        model.push_to_hub(push_path, commit_message=f"Upload {model_name}")
        print("=== Done ===")
    else:
        print("=== Skipping upload to Huggingface Hub ===")

    # upload new README.md to Huggingface Hub
    generate_readme(model_name, results[0], hparams, current_epoch=trainer.current_epoch, model=model)
    if upload_to_hub and hf_entity is not None:
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
        if not upload_to_hub:
            print("    Reason: Upload to Huggingface Hub was disabled.")
        elif hf_entity is None:
            print("    Reason: No Huggingface Hub entity specified.")
        else:
            print("    Reason: Unknown error.")


if __name__ == "__main__":
    typer.run(main)
