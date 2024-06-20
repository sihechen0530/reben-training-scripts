"""
This script loads a pretrained model from the Huggingface Hub and evaluates it on the BigEarthNet v2.0 dataset.
"""
from pathlib import Path
from typing import Optional

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


def download_and_evaluate_model(
        model_name: str = "BIFOLD-BigEarthNetv2-0/BENv2-resnet50-42-s2-v0.1.1",
        limit_test_batches: Optional[int] = 5,
        batch_size: int = 16,
        num_workers_dataloader: int = 8,
):
    model = BENv2ImageEncoder.from_pretrained(model_name)
    model.eval()

    # Load the BigEarthNet v2.0 dataset
    channels = model.config.channels
    image_size = model.config.image_size
    data_dirs = resolve_data_dir(BENv2_DIR_DICT, allow_mock=True)
    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=batch_size,
        num_workers_dataloader=num_workers_dataloader,
        img_size=(channels, image_size, image_size),
    )

    trainer = pl.Trainer(
        accelerator="auto",
        limit_test_batches=limit_test_batches,
    )

    results = trainer.test(model, datamodule=dm)
    print("Finished testing the model.")
    logged_results = {
        "AveragePrecision": {
            "macro": results[0]["test/MultilabelAveragePrecision_macro"],
            "micro": results[0]["test/MultilabelAveragePrecision_micro"],
        },
        "F1Score": {
            "macro": results[0]["test/MultilabelF1Score_macro"],
            "micro": results[0]["test/MultilabelF1Score_micro"],
        },
        "Precision": {
            "macro": results[0]["test/MultilabelPrecision_macro"],
            "micro": results[0]["test/MultilabelPrecision_micro"],
        },
    }
    # print the results as table with one row per metric and one column per metric type
    performance_table = []
    performance_table.append(f"\nResults (in %) for model {model_name}:")
    performance_table.append(f"{'Metric':<16} | Macro | Micro")
    performance_table.append("-" * 33)
    for metric, values in logged_results.items():
        performance_table.append(f"{metric:<16} | {values['macro'] * 100:.2f} | {values['micro'] * 100:.2f}")

    print("\n".join(performance_table))
    return logged_results


if __name__ == "__main__":
    typer.run(download_and_evaluate_model)
