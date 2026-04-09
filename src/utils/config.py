# setup config
download_baseline_model = True
download_audiocaps = ["test"]  # add train and val for training
download_clotho = ["eval"]  # add dev and val for training
browser = "firefox"  # for downloading audiocaps with yt-dlp, e.g. "chrome" or "edge"
browser_cookie_path = ""  # optional path to cookies. for more info see https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp

# evaluation config
baseline = True  # Inference on baseline model
baseline_model = "conette"  # clapcap | conette
dataset = "clotho"  # clotho | audiocaps
metrics = (
    "meteor",
    "spider",
    "fense",
    "vocab",
)
inference = True
save_inference_results = True
evaluation = True
data_folder = "data/"  # Path to data folder
model_folder = "model/"  # Path to model folder
flops_n_samples = 10  # Number of samples to average over for FLOPs measurement

## quantization config
quantization = False  # Inference on quantized model
quantization_mode = "dynamic"  # dynamic | static

## pruning config
pruning = False  # Inference on pruned model
pruning_score_mode = "sum_l2"  # wanda (consider weights and activations) | sum_l2 (consider in and out strength) | first_l2 (consider only in strength)
### conette
decoder_threshold = None
convnext_3072_threshold = 0.075
convnext_1536_threshold = None

### clapcap
gpt_threshold = None  # nach 6 stunden abgebrochen
mapper_threshold = 0.5
htsat_threshold = 0.75
htsat_min_hidden_dim = 1536


## kd config
kd = False  # Inference on saved kd model
kd_model = "best_student_model.pth"  # Path to kd model
patience = 5
num_epochs = 25
batch_size = 32
lr = 1e-5          # decoder + projection learning rate
lr_encoder = 1e-6  # encoder learning rate (lower to avoid overwriting pretrained features)
alpha = 0.5          # KD loss weight (0 = fine-tuning only, 1 = KD only)
temperature = 2.0
kd_save_dir = "checkpoints/kd"

## reproducibility
seed = 42


def set_seed(seed_val: int = None) -> None:
    import random as _random
    import numpy as _np
    import torch as _torch

    s = seed_val if seed_val is not None else seed
    _random.seed(s)
    _np.random.seed(s)
    _torch.manual_seed(s)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(s)


def load_from_yaml(path: str) -> None:
    """Load experiment config from YAML and update this module's globals."""
    import sys
    import yaml

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    g = sys.modules[__name__].__dict__

    # Model & dataset
    if "model" in cfg:
        g["baseline_model"] = cfg["model"]
    if "dataset" in cfg:
        g["dataset"] = cfg["dataset"]

    # Seed
    if "seed" in cfg:
        g["seed"] = cfg["seed"]

    # Pipeline control
    for key in ("inference", "evaluation", "save_inference_results"):
        if key in cfg:
            g[key] = cfg[key]

    # Paths
    if "data_folder" in cfg:
        g["data_folder"] = cfg["data_folder"]
    if "model_folder" in cfg:
        g["model_folder"] = cfg["model_folder"]

    # Metrics
    if "metrics" in cfg:
        g["metrics"] = tuple(cfg["metrics"])

    # technique: none | quantization | pruning | kd | pruning+quantization
    technique = cfg.get("technique", "none")
    g["baseline"] = technique == "none"
    g["quantization"] = "quantization" in technique
    g["pruning"] = "pruning" in technique
    g["kd"] = technique == "kd"

    # Pruning options
    pruning_cfg = cfg.get("pruning", {}) or {}
    if "score_mode" in pruning_cfg:
        g["pruning_score_mode"] = pruning_cfg["score_mode"]
    for key in (
        "decoder_threshold",
        "convnext_3072_threshold",
        "convnext_1536_threshold",
    ):
        if key in pruning_cfg:
            g[key] = pruning_cfg[key]
    for key in (
        "gpt_threshold",
        "mapper_threshold",
        "htsat_threshold",
        "htsat_min_hidden_dim",
    ):
        if key in pruning_cfg:
            g[key] = pruning_cfg[key]

    # FLOPs measurement
    if "flops_n_samples" in cfg:
        g["flops_n_samples"] = cfg["flops_n_samples"]

    # Quantization options
    quant_cfg = cfg.get("quantization", {})
    if isinstance(quant_cfg, dict) and "mode" in quant_cfg:
        g["quantization_mode"] = quant_cfg["mode"]

    # KD options
    kd_cfg = cfg.get("kd", {})
    if isinstance(kd_cfg, dict):
        for key in ("model_path", "num_epochs", "batch_size", "lr", "lr_encoder", "alpha", "temperature", "patience", "save_dir"):
            config_key = {"model_path": "kd_model", "save_dir": "kd_save_dir"}.get(key, key)
            if key in kd_cfg:
                g[config_key] = kd_cfg[key]
