import socket
from pathlib import Path
from typing import Mapping

import lightning.pytorch as pl
import torch
from configilm.extra.BENv2_utils import STANDARD_BANDS
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule
from huggingface_hub import HfApi
from lightning.pytorch.loggers import WandbLogger
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from torchvision import transforms

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

BENv2_DIR_DEFAULT = Path("/projects/SuperResolutionData/sihe.chen/remote_sensing_mock")
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


def get_benv2_dir_dict() -> tuple[str, dict]:
    hostname = socket.gethostname()
    print(f"Using data directories for {hostname}")
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
    else:
        raise ValueError(
            f"Unknown band configuration {bandconfig}, select one of all, s2, s1, rgb or all_full, s2_full, s1_full. The "
            f"full versions include all bands whereas the non-full versions only include the 10m & 20m bands. "
            f"rgb uses 3-channel RGB (B04, B03, B02) from Sentinel-2."
        )
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
):
    # we assume, that we are already logged in to wandb
    if use_wandb:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=True)
    else:
        logger = pl.loggers.WandbLogger(project="BENv2", log_model=False, mode="disabled")

    logger.log_hyperparams(hparams)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/MultilabelAveragePrecision_macro",
        dirpath="./checkpoints",
        filename=f"{hparams['architecture']}-{hparams['seed']}-{hparams['channels']}-val_mAP_macro-" + "{val/MultilabelAveragePrecision_macro:.2f}",
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
        max_epochs=4 if test_run else hparams["epochs"],
        limit_train_batches=4 if test_run else None,
        limit_val_batches=3 if test_run else None,
        limit_test_batches=5 if test_run else None,
        logger=logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, lr_monitor, early_stopping_callback],
    )
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
