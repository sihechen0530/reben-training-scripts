# Training for BigEarthNet v2.0 (reBEN)

## pre-requisites

We recommend using the pyproject.toml file to install the required dependencies based on
[Poetry](https://python-poetry.org/).

Otherwise, the following dependencies are required:

- Python 3.10 or higher (install with `sudo apt install python3.10`)
- configilm[full] (install with `pip install configilm[full]`)
- wandb (install with `pip install wandb`)

Please also create an account on [wandb](https://wandb.ai/) and login using `wandb login`. This is required to log the
training progress.

## Data

The data can be downloaded from the [BigEarthNet website](http://bigearth.net/). Extract the data to a folder and create
an [LMDB](https://lmdb.readthedocs.io/en/release/) database using
the [RSTensorEncoder](https://github.com/kai-tub/rs-tensor-encoder) tool. The tool can be
downloaded from the [BigEarthNet website](http://bigearth.net/) or installed as [AppImage](www.appimage.org) from the
[rs-tensor-encoder](https://github.com/kai-tub/rs-tensor-encoder) repository.

Enter the paths to the following files in the `scripts/train_BENv2.py` script:

- `images_lmdb` (path to the LMDB database)
- `split_csv` (path to the CSV file containing the split information)
- `s1_mapping_csv` (path to the CSV file containing the mapping of the patch IDs to the Sentinel-1 image IDs)
- `labels_csv` (path to the CSV file containing the labels)

Enter the paths at the top of the `scripts/train_BENv2.py` script in the `BENv2_DIR_DICT` dictionary.

## Training

Run the training script with `python scripts/train_BENv2.py`. The script will train the model and log the progress to
wandb.

The following parameters can be adjusted as arguments to the script:

- `--architectures` (default: `resnet18`) The architectures to train. Many architectures from the `timm` library are
  supported.
- `--seed` (default: `42`) The seed to use for the random number generators.
- `--lr` (default: `1e-3`) The learning rate to use for training for the optimizers `AdamW`.
- `--epochs` (default: `100`) The number of epochs to train for.
- `--bs` (default: `16`) The batch size to use for training.
- `--workers` (default: `8`) The number of workers to use for the data loader.
- `--bandconfig` (default: 'all') The band configuration* to use. The following configurations are supported:
    - `all` 10m and 20m bands from Sentinel-2 and all bands from Sentinel-1 (12 bands in total)
    - `s2` 10m and 20m bands from Sentinel-2 (10 bands in total)
    - `s1` all bands from Sentinel-1 (2 bands in total)
    - `all_full` 10m, 20m, and 60m bands from Sentinel-2 and all bands from Sentinel-1 (14 bands in total)
    - `s2_full` 10m, 20m, and 60m bands from Sentinel-2 (12 bands in total)
    - `s1_full` all bands from Sentinel-1 (2 bands in total, same as `s1`)
- `--use-wandb` or `--no-use-wandb` (default: `False`, same as `--no-use-wandb`) Whether to use wandb for logging. If
  `--no-use-wandb` is set, the script will not log to wandb but checkpoints are still saved and metrics are printed to
  the console.
- `--upload-to-hub` or `--no-upload-to-hub` (default: `False`, same as `--no-upload-to-hub`) Whether to upload the model
  to the Hugging Face model hub. If `--upload-to-hub` is set, the model will be uploaded to the Hugging Face model hub
  after training. Note that you have to be logged in to the Hugging Face model hub using `huggingface-cli login` for
  this to work. For this you need a Hugging Face account and a token which can be obtained from the Hugging Face
  website.

*Note: Not all bands from S2 are included in BigEarthNet v2.0. For details, please refer to the
[BigEarthNet v2.0 paper](LINK TODO).