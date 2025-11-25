"""
Utility to plug a self-supervised DINOv3 backbone checkpoint into the
BigEarthNetv2_0_ImageClassifier Lightning module so that it can be used with
``test_checkpoint_BigEarthNetv2_0.py`` (or the regular training scripts).

Example usage (run from scripts/ directory):

python prepare_bigearthnet_classifier_ckpt.py \
    --pretrain-ckpt ../ckpt_logs/dinov3_pretrain_all_14ch_42_xxxx/checkpoints/epoch=99.ckpt \
    --output-ckpt ../ckpt_logs/dinov3_backbone_bootstrap.ckpt \
    --architecture dinov3-small \
    --bandconfig all \
    --dinov3-model-name facebook/dinov3-vits16-pretrain-lvd1689m

The resulting checkpoint can be passed to ``test_checkpoint_BigEarthNetv2_0.py``
or serve as initialization for further supervised fine-tuning.
"""

import sys
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
import typer
from configilm.ConfigILM import ILMConfiguration, ILMType

# Ensure project modules are importable when executed from scripts/
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reben_publication.BigEarthNetv2_0_ImageClassifier import (  # noqa: E402
    BigEarthNetv2_0_ImageClassifier,
)
from scripts.utils import get_bands  # noqa: E402


def _infer_dinov3_name(alias: str) -> str:
    alias_lower = alias.lower()
    if "small" in alias_lower or alias_lower.endswith("s"):
        return "facebook/dinov3-vits16-pretrain-lvd1689m"
    if "base" in alias_lower or alias_lower.endswith("b"):
        return "facebook/dinov3-vitb16-pretrain-lvd1689m"
    if "large" in alias_lower or alias_lower.endswith("l"):
        return "facebook/dinov3-vitl16-pretrain-lvd1689m"
    if "giant" in alias_lower or alias_lower.endswith("g"):
        return "facebook/dinov3-vit7b16-pretrain-lvd1689m"
    # Default to small if unknown
    return "facebook/dinov3-vits16-pretrain-lvd1689m"


def main(
    pretrain_ckpt: str = typer.Option(
        ...,
        help="Path to self-supervised checkpoint produced by train_dinov3_pretraining.py",
    ),
    output_ckpt: str = typer.Option(
        ...,
        help="Path to save the converted BigEarthNet classifier checkpoint (.ckpt)",
    ),
    architecture: str = typer.Option(
        "dinov3-small",
        help="Classifier architecture name (used for bookkeeping / hub loading).",
    ),
    bandconfig: str = typer.Option(
        "all",
        help="Band configuration (must match pre-training).",
    ),
    lr: float = typer.Option(5e-5, help="Learning rate metadata for the classifier."),
    drop_rate: float = typer.Option(0.15, help="Dropout rate metadata for the classifier."),
    drop_path_rate: float = typer.Option(0.0, help="Drop path rate metadata."),
    dinov3_model_name: Optional[str] = typer.Option(
        None,
        help="HuggingFace DINOv3 identifier. If None, inferred from --architecture.",
    ),
    seed: int = typer.Option(42, help="Seed metadata stored in checkpoint."),
):
    assert Path(".").resolve().name == "scripts", (
        "Please run from the scripts directory so relative imports work "
        "(i.e., `cd scripts && python prepare_bigearthnet_classifier_ckpt.py ...`)."
    )

    pl.seed_everything(seed, workers=True)

    if dinov3_model_name is None:
        dinov3_model_name = _infer_dinov3_name(architecture)

    # Determine channels from bandconfig
    _, channels = get_bands(bandconfig)

    # Build ILM configuration and Lightning module
    ilm_config = ILMConfiguration(
        network_type=ILMType.IMAGE_CLASSIFICATION,
        classes=19,
        image_size=120,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        timm_model_name=architecture,
        channels=channels,
    )

    model = BigEarthNetv2_0_ImageClassifier(
        ilm_config,
        lr=lr,
        warmup=None,
        dinov3_model_name=dinov3_model_name,
    )

    # Load pretraining checkpoint (expecting Lightning format with student.* keys)
    ckpt = torch.load(pretrain_ckpt, map_location="cpu")
    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint does not contain a 'state_dict'. Is this a Lightning .ckpt file?")

    student_prefix = "student."
    backbone_prefix = "student.backbone."
    transferred = {}
    for key, value in state_dict.items():
        if key.startswith(backbone_prefix):
            stripped = key[len(student_prefix):]  # e.g., backbone.embeddings...
            transferred[f"model.{stripped}"] = value

    if not transferred:
        raise ValueError(
            "No student.backbone.* weights found in checkpoint. "
            "Ensure you passed a checkpoint created by train_dinov3_pretraining.py."
        )

    load_result = model.load_state_dict(transferred, strict=False)
    print("Loaded backbone weights. Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)

    # Save Lightning-style checkpoint so existing scripts can consume it directly
    from lightning import __version__ as pl_version

    ckpt_payload = {
        "state_dict": model.state_dict(),
        "hyper_parameters": {
            "architecture": architecture,
            "bandconfig": bandconfig,
            "channels": channels,
            "dropout": drop_rate,
            "drop_path_rate": drop_path_rate,
            "seed": seed,
            "lr": lr,
            "dinov3_model_name": dinov3_model_name,
        },
        "pytorch-lightning_version": pl_version,
    }

    output_path = Path(output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_payload, output_path)
    print(f"\nSaved BigEarthNet classifier checkpoint to: {output_path.resolve()}")
    print(
        "You can now run:\n"
        f"  python test_checkpoint_BigEarthNetv2_0.py --checkpoint-path {output_path}\n"
        "to evaluate (assuming you have a trained head) or fine-tune using the "
        "standard training script."
    )


if __name__ == "__main__":
    typer.run(main)

