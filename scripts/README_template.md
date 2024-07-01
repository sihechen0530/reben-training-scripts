---
thumbnail: "https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/RSiM_Logo_1.png"
tags:
    - <MODEL_NAME_RAW>
    - <DATASET_NAME>
    - Remote Sensing
    - Classification
    - image-classification
    - Multispectral
library_name: configilm
license: mit
widget:
  - src: example.png
    example_title: Example
    output:
      - label: <LABEL_1>
        score: <SCORE_1>
      - label: <LABEL_2>
        score: <SCORE_2>
      - label: <LABEL_3>
        score: <SCORE_3>
      - label: <LABEL_4>
        score: <SCORE_4>
      - label: <LABEL_5>
        score: <SCORE_5>
---

[TU Berlin](https://www.tu.berlin/) | [RSiM](https://rsim.berlin/) | [DIMA](https://www.dima.tu-berlin.de/menue/database_systems_and_information_management_group/) | [BigEarth](http://www.bigearth.eu/) | [BIFOLD](https://bifold.berlin/)
:---:|:---:|:---:|:---:|:---:
<a href="https://www.tu.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/tu-berlin-logo-long-red.svg" style="font-size: 1rem; height: 2em; width: auto" alt="TU Berlin Logo"/>  |  <a href="https://rsim.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/RSiM_Logo_1.png" style="font-size: 1rem; height: 2em; width: auto" alt="RSiM Logo"> | <a href="https://www.dima.tu-berlin.de/menue/database_systems_and_information_management_group/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/DIMA.png" style="font-size: 1rem; height: 2em; width: auto" alt="DIMA Logo"> | <a href="http://www.bigearth.eu/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/BigEarth.png" style="font-size: 1rem; height: 2em; width: auto" alt="BigEarth Logo"> | <a href="https://bifold.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/BIFOLD_Logo_farbig.png" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="BIFOLD Logo">

# <MODEL_NAME> pretrained on <DATASET_NAME> using <BANDS_USED> bands

<!-- Optional images -->
<!--
[Sentinel-1](https://sentinel.esa.int/web/sentinel/missions/sentinel-1) | [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)
:---:|:---:
<a href="https://sentinel.esa.int/web/sentinel/missions/sentinel-1"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/sentinel_2.jpg" style="font-size: 1rem; height: 10em; width: auto; margin-right: 1em" alt="Sentinel-2 Satellite"/> | <a href="https://sentinel.esa.int/web/sentinel/missions/sentinel-2"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/sentinel_1.jpg" style="font-size: 1rem; height: 10em; width: auto; margin-right: 1em" alt="Sentinel-1 Satellite"/>
-->

This model was trained on the <DATASET_NAME_FULL_2> dataset using the <BANDS_USED> bands. 
It was trained using the following parameters:
- Number of epochs: up to 100 (with early stopping after 5 epochs of no improvement based on validation average 
precision macro)
- Batch size: 512
- Learning rate: <LEARNING_RATE>
- Dropout rate: <DROPOUT_RATE>
- Drop Path rate: <DROP_PATH_RATE>
- Learning rate scheduler: LinearWarmupCosineAnnealing for <WARMUP_STEPS> warmup steps
- Optimizer: AdamW
- Seed: <SEED>

The weights published in this model card were obtained after <EPOCHS> training epochs.
For more information, please visit the [official <DATASET_NAME_FULL> repository](https://git.tu-berlin.de/rsim/reben-training-scripts), where you can find the training scripts.

![[BigEarthNet](http://bigearth.net/)](https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/combined_2000_600_2020_0_wide.jpg)

The model was evaluated on the test set of the <DATASET_NAME> dataset with the following results:

| Metric            |       Macro |       Micro |
|:------------------|------------------:|------------------:|
| Average Precision |        <AP_MACRO> |        <AP_MICRO> |
| F1 Score          |        <F1_MACRO> |        <F1_MICRO> |
| Precision         | <PRECISION_MACRO> | <PRECISION_MICRO> |

# Example
|             <VIS_BANDS>              |
|:---------------------------------------------------:|
| ![[BigEarthNet](http://bigearth.net/)](example.png) |

| Class labels                                                              |                                                          Predicted scores |
|:--------------------------------------------------------------------------|--------------------------------------------------------------------------:|
| <p> <LABEL_1> <br> <LABEL_2> <br> <LABEL_3> <br> ... <br> <LABEL_19> </p> | <p> <SCORE_1> <br> <SCORE_2> <br> <SCORE_3> <br> ... <br> <SCORE_19> </p> |


To use the model, download the codes that define the model architecture from the
[official <DATASET_NAME_FULL> repository](https://git.tu-berlin.de/rsim/reben-training-scripts) and load the model using the
code below. Note that you have to install [`configilm`](https://pypi.org/project/configilm/) to use the provided code.

```python
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

model = BigEarthNetv2_0_ImageClassifier.from_pretrained("path_to/huggingface_model_folder")
```

e.g.

```python
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
  "BIFOLD-BigEarthNetv2-0/<MODEL_NAME_RAW>-<BAND_CONFIG>-v0.1.1")
```

If you use this model in your research or the provided code, please cite the following papers:
```bibtex
CITATION FOR DATASET PAPER
```
```bibtex
@article{hackel2024configilm,
  title={ConfigILM: A general purpose configurable library for combining image and language models for visual question answering},
  author={Hackel, Leonard and Clasen, Kai Norman and Demir, Beg{\"u}m},
  journal={SoftwareX},
  volume={26},
  pages={101731},
  year={2024},
  publisher={Elsevier}
}
```
