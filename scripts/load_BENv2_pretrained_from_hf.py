"""
This script loads a pretrained model from the Huggingface Hub and evaluates it on the BigEarthNet v2.0 dataset.
"""
from pathlib import Path

import lightning.pytorch as pl
import typer
from configilm.extra.BENv2_utils import resolve_data_dir
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule

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
        model_name: str = "hackelle/BENv2-resnet18-42-all-v0.1.1-alpha",
):
    model = BENv2ImageEncoder.from_pretrained(model_name)
    model.eval()

    # Load the BigEarthNet v2.0 dataset
    channels = 12
    data_dirs = resolve_data_dir(BENv2_DIR_DICT, allow_mock=True)
    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=16,
        num_workers_dataloader=8,
        img_size=(channels, 120, 120),
    )

    trainer = pl.Trainer(
        accelerator="auto",
        limit_test_batches=20,
    )

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    typer.run(main)
