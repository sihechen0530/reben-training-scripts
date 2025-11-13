"""
This script loads a pretrained model from the Huggingface Hub and evaluates it on the BigEarthNet v2.0 dataset.
"""
import sys
from pathlib import Path
from typing import Optional
import difflib

# Add parent directory to path to allow importing reben_publication
# This allows running from the scripts directory
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import lightning.pytorch as pl
import typer
from huggingface_hub import HfApi
from configilm.extra.BENv2_utils import resolve_data_dir
from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
from scripts.utils import get_benv2_dir_dict

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def download_and_evaluate_model(
        model_name: str = "hackelle/resnet18-all-v0.1.1",
        limit_test_batches: Optional[int] = 4,
        batch_size: int = 32,
        num_workers_dataloader: int = 8,
):
    api = HfApi()
    if not api.repo_exists(model_name):
        entity = model_name.split("/")[0]
        models = api.list_models(author=entity)
        models = [x.id for x in models]
        most_similar = difflib.get_close_matches(model_name, models)
        if most_similar == []:
            assert False, (f"Model {model_name} does not exist in the Huggingface Hub. No similarly named models found."
                           f"Maybe the user/org {entity} does not exist?")
        assert False, f"Model {model_name} does not exist in the Huggingface Hub. Did you mean one of {most_similar}?"
    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_name)
    model.eval()

    # Load the BigEarthNet v2.0 dataset
    channels = model.config.channels
    image_size = model.config.image_size
    hostname, data_dirs = get_benv2_dir_dict()
    data_dirs = resolve_data_dir(data_dirs, allow_mock=True)
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
