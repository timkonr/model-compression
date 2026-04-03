"""
mc-run-all: Run the full experiment matrix defined in a YAML file.

Usage:
    mc-run-all --config experiments/full_matrix.yaml
    mc-run-all --config experiments/full_matrix.yaml --dry-run

Matrix YAML format:
    defaults:
      seed: 42
      metrics: [spider, fense, meteor]
      data_folder: data/
      model_folder: model/
      inference: true
      evaluation: true
      save_inference_results: true

    experiments:
      - model: conette
        dataset: clotho
        technique: none
      - model: conette
        dataset: clotho
        technique: quantization
      ...
"""

import argparse
import subprocess
import sys
import tempfile
import os
import yaml


def merge(defaults: dict, experiment: dict) -> dict:
    """Shallow merge: experiment values override defaults."""
    merged = dict(defaults)
    for k, v in experiment.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def run_experiment(cfg: dict, dry_run: bool) -> int:
    """Write cfg to a temp YAML and call mc-evaluate. Returns exit code."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="mc_exp_"
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False)
        tmp_path = f.name

    label = f"{cfg.get('model','?')} | {cfg.get('dataset','?')} | {cfg.get('technique','none')}"
    pruning_cfg = cfg.get("pruning", {})
    if pruning_cfg:
        ratios = {k: v for k, v in pruning_cfg.items() if "keep_ratio" in k and v is not None}
        if ratios:
            label += f" | {ratios}"

    print(f"\n{'='*70}")
    print(f"  Experiment: {label}")
    print(f"  Config:     {tmp_path}")
    print(f"{'='*70}")

    if dry_run:
        print("  [dry-run] skipping execution")
        os.unlink(tmp_path)
        return 0

    try:
        result = subprocess.run(
            [sys.executable, "-m", "evaluate", "--config", tmp_path],
            check=False,
        )
        return result.returncode
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment matrix.")
    parser.add_argument("--config", required=True, help="Path to full_matrix.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiments without running them",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        default=True,
        help="Continue to next experiment if one fails (default: True)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        matrix = yaml.safe_load(f)

    defaults = matrix.get("defaults", {})
    experiments = matrix.get("experiments", [])

    if not experiments:
        print("No experiments found in matrix config.")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments.")

    failed = []
    for i, exp in enumerate(experiments):
        cfg = merge(defaults, exp)
        print(f"\n[{i+1}/{len(experiments)}]", end="")
        rc = run_experiment(cfg, dry_run=args.dry_run)
        if rc != 0:
            label = f"{cfg.get('model')} | {cfg.get('dataset')} | {cfg.get('technique')}"
            failed.append(label)
            if not args.skip_errors:
                print(f"\nExperiment failed (exit code {rc}). Stopping.")
                sys.exit(rc)
            else:
                print(f"  [WARNING] Experiment failed (exit code {rc}), continuing.")

    print(f"\n{'='*70}")
    print(f"Done. {len(experiments) - len(failed)}/{len(experiments)} experiments succeeded.")
    if failed:
        print("Failed experiments:")
        for f in failed:
            print(f"  - {f}")
    print(f"{'='*70}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
