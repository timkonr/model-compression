"""Lightweight bs=1 inference-latency benchmark for a single CoNeTTE model config.

Meant for resource-constrained hardware (Raspberry Pi). This script reuses load_model()
and the exact same measure_latency() logic as evaluate.py (same latency_n_samples /
latency_n_warmup config fields) so numbers stay directly comparable to the GPU runs
in eval.json, but skips dataset-wide inference and aac_metrics entirely.

Usage: python src/measure_latency_pi.py --config path/to/config.yaml [--out latency.json]
"""

import argparse
import json
import os
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from aac_datasets import Clotho, AudioCaps
from aac_datasets.utils.collate import BasicCollate

from utils.utils import load_model
from utils.model_size import get_model_size, get_model_params
from utils import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure bs=1 inference latency for a single model config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a YAML experiment config file."
    )
    parser.add_argument(
        "--out",
        help="Path to save the latency result JSON "
        "(default: latency_pi[_quantized].json next to --config).",
    )
    return parser.parse_args()


def load_dataset(subset, dataset):
    if dataset == "clotho":
        return Clotho(config.data_folder, subset=subset)
    return AudioCaps(config.data_folder, subset=subset, audio_format="wav", sr=22050)


def build_calib_loader():
    """Same logic as evaluate.py's perform_inference(): only needed for live pruning
    (technique: pruning/kd with score_mode=wanda). Checkpoint-loading kd configs
    (kd.model_path pointing at a saved, already-pruned student) don't need this."""
    if not (config.pruning_score_mode == "wanda" and (config.pruning or config.kd)):
        return None
    calib_dataset = getattr(config, "pruning_calibration_dataset", None) or config.dataset
    calib_subset = "val" if calib_dataset == "audiocaps" else "dev"
    ds = load_dataset(calib_subset, calib_dataset)
    shuffle = getattr(config, "pruning_calibration_shuffle", False)
    generator = torch.Generator().manual_seed(config.seed) if shuffle else None
    return DataLoader(
        ds, batch_size=1, collate_fn=BasicCollate(), shuffle=shuffle, generator=generator
    )


def measure_latency(model, dataset, device, n_samples, n_warmup):
    n = min(n_samples + n_warmup, len(dataset))
    loader = DataLoader(
        torch.utils.data.Subset(dataset, list(range(n))),
        batch_size=1,
        collate_fn=BasicCollate(),
    )
    model.eval()
    is_cuda = device.startswith("cuda")
    times = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if is_cuda:
                torch.cuda.synchronize()
            start = perf_counter()
            model(batch["audio"], batch["sr"], task=config.dataset)
            if is_cuda:
                torch.cuda.synchronize()
            elapsed = perf_counter() - start
            if i >= n_warmup:
                times.append(elapsed)
    times = torch.tensor(times)
    return {
        "latency_ms_per_sample_mean": (times.mean() * 1000).item(),
        "latency_ms_per_sample_std": (
            (times.std(unbiased=False) * 1000).item() if len(times) > 1 else 0.0
        ),
        "n_measured": len(times),
    }


def technique_name():
    if config.kd and config.quantization:
        return "kd+quantization"
    if config.pruning and config.quantization:
        return "pruning+quantization"
    if config.pruning:
        return "pruning"
    if config.quantization:
        return "quantization"
    if config.kd:
        return "kd"
    return "baseline"


def main():
    args = parse_args()
    config.load_from_yaml(args.config)
    config.set_seed(config.seed)

    if config.baseline_model != "conette":
        raise ValueError(
            "This script only supports CoNeTTE (relies on model(audio, sr, task=...))."
        )

    technique = technique_name()
    print(f"loading model: technique={technique}, dataset={config.dataset}")
    calib_loader = build_calib_loader()
    model = load_model(quantized=config.quantization, pruned=config.pruning, kd=config.kd, loader=calib_loader)

    device = str(next(model.parameters()).device)
    model_size_mb = get_model_size(model)
    model_params = get_model_params(model)

    subset = "test" if config.dataset == "audiocaps" else "eval"
    test_ds = load_dataset(subset, config.dataset)

    print(
        f"measuring latency (bs=1, n={config.latency_n_samples}, "
        f"warmup={config.latency_n_warmup}, device={device})..."
    )
    latency_stats = measure_latency(
        model, test_ds, device, config.latency_n_samples, config.latency_n_warmup
    )

    result = {
        "model": config.baseline_model,
        "compression_technique": technique,
        "dataset": config.dataset,
        "seed": config.seed,
        "device": device,
        "model_size_mb": model_size_mb,
        "unquantized_parameters": model_params,
        **latency_stats,
    }
    print(json.dumps(result, indent=2))

    out_path = args.out
    if not out_path:
        exp_dir = os.path.dirname(os.path.abspath(args.config))
        suffix = "_quantized" if config.quantization else ""
        out_path = os.path.join(exp_dir, f"latency_pi{suffix}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
