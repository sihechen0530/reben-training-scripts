# Training for BigEarthNet v2.0 (reBEN)

<a href="https://www.tu.berlin/"><img src="_res/img/logos/TU-Berlin.svg" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="TU Berlin Logo"/>
<img height="2em" hspace="10em"/>
<a href="https://rsim.berlin/"><img src="_res/img/logos/RSiM.png" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="RSiM Logo"/>
<img height="2em" hspace="10em"/>
<a href="https://www.dima.tu-berlin.de/menue/database_systems_and_information_management_group/"><img src="_res/img/logos/DIMA.png" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="DIMA Logo"/>
<img height="2em" hspace="10em"/>
<a href="http://bigearth.net/"><img src="_res/img/logos/BigEarth.png" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="BigEarth Logo"/>
<img height="2em" hspace="10em"/>
<a href="https://bifold.berlin/"><img src="_res/img/logos/bifold.png" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="BIFOLD Logo"/>

[![Dataset](https://img.shields.io/badge/Dataset%20on-Zenodo-014baa.svg)](LINK PENDING)
[![Publication arXiv](https://img.shields.io/badge/Publication%20on-arXiv-b21a1a.svg)](LINK PENDING)
[![Publication](https://img.shields.io/badge/Publication%20on-IEEE%20Xplore-103546.svg)](LINK PENDING)

<a href="http://bigearth.net/"><img src="_res/img/combined_2000_600_2020_0.jpg" style="font-size: 1rem; height: 11.3em; width: auto; margin-right: 1em" alt="BigEarth Logo"/>
## pre-requisites

We recommend using the pyproject.toml file to install the required dependencies based on
[Poetry](https://python-poetry.org/). For use with Poetry, CUDA 12.x is required. To use CUDA 11.x, the commented lines
in the pyproject.toml file have to be uncommented. To install the dependencies using Poetry, run `poetry install`.

Otherwise, the following dependencies are required:

- Python 3.9 - 3.12 (e.g. install with `sudo apt install python3.10`)
- configilm[full] (e.g. install with `pip install configilm[full]~=0.6.5`) 0.6.5 or higher
- wandb (e.g. install with `pip install wandb`)
- numpy 1.x (e.g. install with `pip install numpy~=1.26.4`)

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
- `--drop_rate` (default: `0.375`) The dropout rate to use for the models.
- `--drop_path_rate` (default: `0.0`) The drop path rate to use for the models.
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
- `--test-run` or `--no-test-run` (default: `True`) Whether to only run a few batches for testing. If `--no-test-run` is
  set, the full dataset will be used for training and testing.

The full command to train the resnet50 model as described in the BigEarthNet v2.0 paper with all bands from Sentinel-2 
is as follows:

`python train_BENv2.py --no-test-run --use-wandb --upload-to-hub --architecture=resnet50 --bandconfig=s2 --bs=512 --lr=0.001`

This command
 - used the full dataset instead of only a few batches to test
 - logged the training progress to wandb
 - uploaded the model to the Hugging Face model hub after training and testing
 - used the resnet50 architecture
 - used only the Sentinel-2 bands
 - used a batch size of 512
 - used a learning rate of 0.001

The trained model will be saved in the `models` directory and on huggingface. It can be loaded using
```python
from ben_publication.BENv2ImageClassifier import BENv2ImageEncoder

model = BENv2ImageEncoder.from_pretrained("<entity>/<model-name>")
```
e.g.
```python
from ben_publication.BENv2ImageClassifier import BENv2ImageEncoder

model = BENv2ImageEncoder.from_pretrained("BIFOLD-BigEarthNetv2-0/BENv2-resnet50-42-s2-v0.1.1")
```
Note, that the model configuration is automatically loaded from the Hugging Face model hub.

*Note: Not all bands from S2 are included in BigEarthNet v2.0. For details, please refer to the
[BigEarthNet v2.0 paper](LINK TODO).