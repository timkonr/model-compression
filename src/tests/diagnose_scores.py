"""
Diagnostic script: inspect activation-based importance scores on the real CoNeTTE model.

Purpose: determine whether wanda scores have meaningful signal (high variance,
clear separation between important and unimportant neurons) or whether all
neurons look equally important (CV ≈ 0, top/bottom ratio ≈ 1x → random ≈ wanda).

Usage (from repo root):
    python src/tests/diagnose_scores.py \
        --model_path model/baseline/ \
        --dataset clotho \
        --data_folder data/ \
        --num_batches 10

Outputs:
    - Confirmation that hooks fired (num_samples > 0)
    - Per-layer score statistics for wanda, l2, and random
    - Signal ratio: top-10% mean / bottom-10% mean (1.0x = no signal, 10x = strong signal)
    - Overlap between wanda and random keep-sets at 50% pruning ratio
"""

import sys
import os
import argparse
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from aac_datasets import Clotho, AudioCaps
from aac_datasets.utils.collate import BasicCollate
from conette import CoNeTTEConfig, CoNeTTEModel
from torch.utils.data import DataLoader

from prune import (
    ActivationCollector,
    collect_conette_encoder_activation_scores,
    collect_conette_decoder_activation_scores,
    compute_linear_hidden_scores,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _signal_ratio(scores: torch.Tensor, top_fraction: float = 0.1) -> float:
    """
    top_fraction mean / bottom_fraction mean.

    1.0x → no discriminative signal (all neurons look equally important).
    Higher → clearer separation (wanda can distinguish important from unimportant).
    """
    k = max(1, int(len(scores) * top_fraction))
    top_mean = scores.topk(k, largest=True).values.mean().item()
    bot_mean = scores.topk(k, largest=False).values.mean().item()
    if bot_mean < 1e-9:
        return float("inf")
    return top_mean / bot_mean


def _keep_set_overlap(scores_a: torch.Tensor, scores_b: torch.Tensor, keep_ratio: float = 0.5) -> float:
    """
    Fraction of neurons that both scoring methods agree to keep.

    1.0 = perfect agreement, 0.0 = complete disagreement.
    Random baseline: ~keep_ratio (e.g. 0.5 for 50% pruning).
    """
    k = max(1, int(len(scores_a) * keep_ratio))
    keep_a = set(scores_a.topk(k, largest=True).indices.tolist())
    keep_b = set(scores_b.topk(k, largest=True).indices.tolist())
    return len(keep_a & keep_b) / k


def _print_layer_stats(name: str, scores: torch.Tensor) -> None:
    n = len(scores)
    mn, mx = scores.min().item(), scores.max().item()
    mean = scores.mean().item()
    std = scores.std().item()
    cv = std / (mean + 1e-12)
    sig = _signal_ratio(scores)
    print(
        f"    n={n:4d}  min={mn:.4f}  max={mx:.4f}  "
        f"mean={mean:.4f}  std={std:.4f}  CV={cv:.3f}  "
        f"top10%/bot10%={sig:.2f}x"
    )


def _compare_to_random_and_l2(
    name: str,
    wanda_scores: torch.Tensor,
    first: nn.Linear,
    second: nn.Linear,
    keep_ratio: float = 0.5,
) -> None:
    """Print agreement between wanda and alternative scoring methods."""
    torch.manual_seed(0)
    random_scores = compute_linear_hidden_scores(first, second, mode="random")
    l2_scores = compute_linear_hidden_scores(first, second, mode="first_l2")

    wanda_vs_random = _keep_set_overlap(wanda_scores, random_scores, keep_ratio)
    wanda_vs_l2 = _keep_set_overlap(wanda_scores, l2_scores, keep_ratio)
    random_vs_l2 = _keep_set_overlap(random_scores, l2_scores, keep_ratio)

    expected_random = keep_ratio  # random agreement with anything ≈ keep_ratio
    print(
        f"    keep-set agreement at {keep_ratio:.0%} keep ratio:\n"
        f"      wanda vs random = {wanda_vs_random:.2%}  "
        f"(expected if no signal: ~{expected_random:.0%})\n"
        f"      wanda vs l2    = {wanda_vs_l2:.2%}\n"
        f"      random vs l2   = {random_vs_l2:.2%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic
# ─────────────────────────────────────────────────────────────────────────────


def diagnose(
    model_path: str,
    dataset: str,
    data_folder: str,
    num_batches: int,
    batch_size: int = 4,
    keep_ratio: float = 0.5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Score Diagnostic — CoNeTTE")
    print(f"  model:    {model_path}")
    print(f"  dataset:  {dataset}")
    print(f"  batches:  {num_batches}  (batch_size={batch_size})")
    print(f"  device:   {device}")
    print(f"{'='*70}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model = CoNeTTEModel.from_pretrained(
        model_path, config=CoNeTTEConfig.from_pretrained(model_path)
    )
    model.to(device)
    model.eval()
    print("  OK\n")

    # ── Build calibration loader (training split) ──────────────────────────
    print("Building calibration loader...")
    subset = "train" if dataset == "audiocaps" else "dev"
    if dataset == "clotho":
        ds = Clotho(data_folder, subset=subset)
    elif dataset == "audiocaps":
        ds = AudioCaps(data_folder, subset=subset, audio_format="wav", sr=22050)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=BasicCollate(),
        num_workers=0,
    )
    print(f"  {len(ds)} samples, using first {num_batches} batches\n")

    # ── Collect encoder scores ────────────────────────────────────────────────
    print("─" * 70)
    print("ENCODER ACTIVATION SCORES (pwconv1/pwconv2 pairs)")
    print("─" * 70)

    encoder_scores = collect_conette_encoder_activation_scores(
        model, loader=loader, num_batches=num_batches, task=dataset
    )

    if not encoder_scores:
        print("  ERROR: No encoder scores collected — hooks did not fire!")
    else:
        for name, scores in sorted(encoder_scores.items()):
            layer_key = name.replace(".pwconv1", "")
            # Retrieve the actual linear pair for comparison
            parts = layer_key.split(".")
            stage_idx, block_idx = int(parts[3]), int(parts[4])
            block = model.preprocessor.encoder.stages[stage_idx][block_idx]

            print(f"\n  {name}")
            print(f"  wanda:")
            _print_layer_stats(name, scores)

            _compare_to_random_and_l2(
                name, scores, block.pwconv1, block.pwconv2, keep_ratio
            )

    # ── Collect decoder scores ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("DECODER ACTIVATION SCORES (linear1/linear2 pairs)")
    print("─" * 70)

    decoder_scores = collect_conette_decoder_activation_scores(
        model, loader=loader, num_batches=num_batches, dataset_name=dataset
    )

    if not decoder_scores:
        print("  ERROR: No decoder scores collected — hooks did not fire!")
    else:
        for name, scores in sorted(decoder_scores.items()):
            li = int(name.split(".")[3])
            layer = model.model.decoder.layers[li]

            print(f"\n  {name}")
            print(f"  wanda:")
            _print_layer_stats(name, scores)

            _compare_to_random_and_l2(
                name, scores, layer.linear1, layer.linear2, keep_ratio
            )

    # ── Global summary ────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("SUMMARY")
    print("─" * 70)

    all_scores = list(encoder_scores.values()) + list(decoder_scores.values())
    if all_scores:
        all_cvs = [s.std().item() / (s.mean().item() + 1e-12) for s in all_scores]
        all_sigs = [_signal_ratio(s) for s in all_scores]
        print(f"  Layers diagnosed:      {len(all_scores)}")
        print(f"  Mean CV across layers: {sum(all_cvs)/len(all_cvs):.3f}")
        print(f"  Mean signal ratio:     {sum(all_sigs)/len(all_sigs):.2f}x")
        print()
        print("  Interpretation:")
        mean_cv = sum(all_cvs) / len(all_cvs)
        mean_sig = sum(all_sigs) / len(all_sigs)
        if mean_cv < 0.05 or mean_sig < 1.5:
            print("  → LOW signal. All neurons look equally important.")
            print("    wanda ≈ random is expected. This is a property of the model,")
            print("    not a bug. CoNeTTE may not have redundant neurons.")
        elif mean_cv < 0.2 or mean_sig < 3.0:
            print("  → MODERATE signal. Some discrimination possible.")
            print("    Check per-layer stats above for which layers have the most signal.")
        else:
            print("  → HIGH signal. wanda should clearly outperform random.")
            print("    If it doesn't, investigate the pruning application.")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose activation-based pruning scores.")
    p.add_argument("--model_path", default="model/baseline/", help="Path to CoNeTTE model")
    p.add_argument("--dataset", default="clotho", choices=["clotho", "audiocaps"])
    p.add_argument("--data_folder", default="data/", help="Path to dataset root")
    p.add_argument("--num_batches", type=int, default=10, help="Calibration batches")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--keep_ratio", type=float, default=0.5, help="Keep ratio for overlap analysis")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    diagnose(
        model_path=args.model_path,
        dataset=args.dataset,
        data_folder=args.data_folder,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        keep_ratio=args.keep_ratio,
    )
