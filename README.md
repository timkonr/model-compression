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

First you have to clone this repository.

```bash
git clone https://github.com/timkonr/model-compression.git
cd model-compression
```

Afterwards you need to create an environment that contains **python>=3.12** and **pip**. You can use venv, conda, micromamba or some other python environment tool.

Here is an example using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):

```bash
conda create -n model_compression python=3.12.7
conda activate model_compression
```

Then, you can install the required packages for this repository:

```bash
pip install -r requirements.txt -c constraints.txt
pip install -e .
```

You also need to install Java >= 1.8 and <= 1.13 on your machine to compute AAC metrics.

If you intend to use the AudioCaps dataset, **ffmpeg** and **yt-dlp** need to be installed as well. See [aac-datasets](https://aac-datasets.readthedocs.io/en/stable/installation.html#external-requirements-audiocaps-only) for more information.

Note: It might be necessary to install the proper pytorch version with CUDA manually.

### Download external data, models and prepare

To download, extract and process data, you need to run:

```bash
mc-prepare
```

By default, the dataset is stored in the `./data` directory and the pre-trained CoNeTTE model is stored in the `./model` directory.

Next, you need to download the required files for the metrics.

```bash
aac-metrics-download
```

## Usage

### Evaluate

For evaluation, the package [aac-metrics](https://aac-metrics.readthedocs.io) is used.
To run evaluations on a model use the command:

```bash
mc-evaluate
```

Configuration of the evaluation can be managed in the config.py.

### Results

After successful inference of the dataset by the selected models, a JSON file is saved in the results folder called `inference_results_{compression_technique}_{timestamp}.json`. It includes the model and compression technique used, model size in MB, the amount of unquantized parameters, whether the device used was GPU or CPU, total inference time in seconds, additional pruning config if using a pruned model, the generated captions and its corresponding baseline references.

Next, selected metrics are calculated using the generated and baseline captions and saved together with the above mentioned metadata as `eval_results_{device}_{baseline/quantized}_{timestamp}.json`.
