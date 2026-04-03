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

Experiments are configured via YAML files in `experiments/`. No manual code editing required.

`mc-run` accepts any YAML — it auto-detects whether it describes a single experiment or a full matrix:

```bash
mc-run --config experiments/example_single.yaml        # single experiment
mc-run --config experiments/full_matrix.yaml           # full matrix
mc-run --config experiments/smoke_test.yaml            # smoke test (one per technique)
mc-run --config experiments/full_matrix.yaml --dry-run # preview without running
```

In matrix mode, each experiment runs in its own subprocess so memory is fully released between runs.

Edit `experiments/example_single.yaml` to configure an experiment. The available techniques are:

| `technique` value | Description |
|---|---|
| `none` | Uncompressed baseline |
| `quantization` | Dynamic INT8 quantization |
| `pruning` | Structured pruning (L2-norm or random) |
| `kd` | Load a KD-trained student model |
| `pruning+quantization` | Pruning followed by quantization |

### Train a KD student model

After pruning, a student model can be recovered via knowledge distillation:

```bash
mc-train-kd --student-path checkpoints/pruned/ --dataset audiocaps --epochs 10
```

The teacher is always the unpruned baseline loaded from `model/baseline/`.

### Results

Each experiment saves two JSON files to `results/`:

- `inference_results_{technique}_{timestamp}.json` — generated captions, references, model size, inference time, pruning config
- `eval_results_{device}_{technique}_{timestamp}.json` — metric scores (SPIDEr, FENSE, METEOR) plus all metadata
