"""
This script loads a pretrained model from the Huggingface Hub and evaluates it on a random input.
"""

import torch
from huggingface_hub import HfApi

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


def download_and_evaluate_model(
        model_name: str = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0",
        batch_size: int = 4,
):
    # Check if the model exists in the Huggingface Hub
    api = HfApi()
    assert api.repo_exists(model_name), f"Model {model_name} does not exist in the Huggingface Hub."

    # Load the model
    model = BigEarthNetv2_0_ImageClassifier.from_pretrained(model_name)
    model.eval()

    # Test the model with a random input
    channels = model.config.channels
    image_size = model.config.image_size

    x = torch.randn(batch_size, channels, image_size, image_size)
    print("Input: ", x.shape)
    y = model(x)
    assert y.shape[0] == batch_size, f"Expected batch size {batch_size}, got {y.shape[0]}."
    print("Output: ", y.shape)


if __name__ == "__main__":
    download_and_evaluate_model()
