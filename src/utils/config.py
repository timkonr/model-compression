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
num_calibration_batches = 128  # Number of batches to use for collecting activation stats for pruning (only for wanda score mode)
### conette
decoder_pruning_ratio = None
convnext_3072_threshold = 0.075
convnext_1536_threshold = None
global_pruning_ratio = (
    None  # if set: global pruning mode (overrides convnext thresholds)
)

### clapcap
htsat_pruning_ratio = None   # global ratio across all HTSAT MLP blocks
mapper_pruning_ratio = None  # global ratio across all Mapper MLP layers
# GPT-2 decoder excluded: structured pruning caused 6x inference slowdown


## kd config
kd = False  # Inference on saved kd model
kd_model = "best_student_model.pth"  # Path to kd model
patience = 5
num_epochs = 25
batch_size = 32
lr = 1e-5  # decoder + projection learning rate
grad_accum_steps = (
    1  # gradient accumulation steps (effective_batch = batch_size * grad_accum_steps)
)
lr_encoder = (
    1e-6  # encoder learning rate (lower to avoid overwriting pretrained features)
)
kd_mode = "pure_kd"  # pure_kd (Minitron BP #5) | hybrid (Hinton: α·CE + (1-α)·KD) | encoder_ce (CE only)
kd_alpha = 0.5  # hybrid only: weight on CE loss (0=pure KD, 1=pure CE)
kd_train_components = "all"  # encoder | all (encoder + decoder + projection)
kd_save_dir = "checkpoints/kd"
weight_decay = 1e-5
grad_clip_norm = 1.0

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
        "decoder_pruning_ratio",
        "convnext_3072_threshold",
        "convnext_1536_threshold",
        "global_pruning_ratio",
        "num_calibration_batches",
    ):
        if key in pruning_cfg:
            g[key] = pruning_cfg[key]
    for key in (
        "htsat_pruning_ratio",
        "mapper_pruning_ratio",
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
        for key in (
            "model_path",
            "num_epochs",
            "batch_size",
            "grad_accum_steps",
            "lr",
            "lr_encoder",
            "patience",
            "save_dir",
            "mode",
            "alpha",
            "train_components",
            "weight_decay",
            "grad_clip_norm",
        ):
            config_key = {
                "model_path": "kd_model",
                "save_dir": "kd_save_dir",
                "mode": "kd_mode",
                "alpha": "kd_alpha",
                "train_components": "kd_train_components",
            }.get(key, key)
            if key in kd_cfg:
                g[config_key] = kd_cfg[key]
