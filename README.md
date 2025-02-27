# model-compression

<div align="center">

**Low-Complexity Automated Audio Captioning**

<a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/-Python 3.11-blue?style=for-the-badge&logo=python&logoColor=white">
</a>
<a href="https://pytorch.org/get-started/locally/">
    <img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white">
</a>
<a href="https://black.readthedocs.io/en/stable/">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray">
</a>

</div>

The aim of this repository is to provide a platform for experimenting with various model compression techniques on an AAC model with competitive performance on the state-of-the-art as part of my master's thesis. In this case, the model I opted for is [CoNeTTE](https://github.com/Labbeti/conette-audio-captioning), which also served in a slightly modified form as the baseline model for the DCASE 2024 challenge in AAC.

## Installation
First, you need to create an environment that contains **python>=3.11** and **pip**. You can use venv, conda, micromamba or some other python environment tool.

Here is an example with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
```bash
conda create -n model_compression python=3.11
conda activate model_compression
```

Then, you can clone this repository and install it:
```bash
git clone https://github.com/timkonr/model-compression.git
cd model-compression
pip install -e .
```

You also need to install Java >= 1.8 and <= 1.13 on your machine to compute AAC metrics.

If you intend to use the AudioCaps dataset, **ffmpeg** and **yt-dlp** need to be installed as well. See [aac-datasets](https://aac-datasets.readthedocs.io/en/stable/installation.html#external-requirements-audiocaps-only) for more information.

Note: It might be necessary to install the proper pytorch version with CUDA manually.

## Usage

### Download external data, models and prepare

To download, extract and process data, you need to run:
```bash
mc-prepare
```
By default, the dataset is stored in the `./data` directory and the pre-trained CoNeTTE model is stored in the `./model` directory.

### Evaluate

For evaluation, the package [aac-metrics](https://aac-metrics.readthedocs.io) is used.
To run evaluations on a model use the command:
```bash
mc-evaluate
```

Which model to evaluate (and what metrics to use - TODO) can be managed by adding command line args.
By default, only the baseline model is evaluated and the metrics used are:
```json
[
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "meteor",
    "rouge_l",
    "fense",
    "spider_fl",
    "vocab",
    "bert_score"
]
```

Additional info collected is the model size in MB, the device used for inference and the inference time.

### Visualizations

TODO

## TODOS
- [x] project set up
    - [x] GitHub (repo, readme)
    - [x] download dataset/model
    - [x] requirements.txt
    - [x] add config
    - [x] add pyproject.toml
    - [x] add config for choosing metrics
    - [x] allow evaluation for different datasets
    - [x] add timer for inference
    - [x] log device used in evaluation
    - [x] separate inference from evaluation
        - [x] allow for using previous inference results in evaluation
- [ ] quantization
    - [x] set up quantization
    - [ ] compare different quantization methods
    - [x] compare inference time on cpu for baseline model
- [ ] knowledge distillation
    - [ ] set up pipeline
    - [ ] experiment with model architecture and hyperparameters
    - [ ] fine-tuning pipeline
- [ ] pruning
    - [x] set up pruning
    - [ ] compare various pruning settings ((globally) unstructured, structured)
    - [ ] log pruning params
    - [ ] experiment with sparse models
    - [ ] fine-tuning pipeline
- [ ] visualizations
    - [ ] line charts comparing performance over various compression levels
    - [ ] charts comparing model sizes/inference time/memory usage
    - [ ] scatter plot comparing performance/model efficiency