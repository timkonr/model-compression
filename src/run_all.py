"""
mc-run: Run one experiment or a full matrix, depending on the YAML config.

Single experiment (has 'model' key):
    mc-run --config experiments/example_single.yaml

Full matrix (has 'experiments' key):
    mc-run --config experiments/full_matrix.yaml [--dry-run] [--fail-fast]
"""

import argparse
import subprocess
import sys
import tempfile
import os
import yaml


def _is_matrix(cfg: dict) -> bool:
    return "experiments" in cfg


def merge(defaults: dict, experiment: dict) -> dict:
    merged = dict(defaults)
    for k, v in experiment.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


_CONETTE_PRUNING_KEYS = {"decoder_keep_ratio", "convnext_3072_keep_ratio", "convnext_1536_keep_ratio"}
_CLAPCAP_PRUNING_KEYS = {"gpt_keep_ratio", "mapper_keep_ratio", "htsat_keep_ratio", "htsat_min_hidden_dim"}


def _label(cfg: dict) -> str:
    return f"{cfg.get('model', '?')} | {cfg.get('dataset', '?')} | {cfg.get('technique', 'none')} | seed: {cfg.get('seed', '?')}"


def _print_header(cfg: dict):
    print(f"\n{'='*70}")
    print(f"  {_label(cfg)}")
    pruning_cfg = cfg.get("pruning") or {}
    if cfg.get("technique") == "pruning" and pruning_cfg:
        model = cfg.get("model", "")
        relevant_keys = _CONETTE_PRUNING_KEYS if model == "conette" else _CLAPCAP_PRUNING_KEYS
        print(f"  score_mode: {pruning_cfg.get('score_mode', '?')}")
        for k, v in pruning_cfg.items():
            if k in relevant_keys and v is not None:
                print(f"  {k}: {v}")
    print(f"{'='*70}")


def run_single_subprocess(cfg: dict, dry_run: bool) -> int:
    """Write cfg to a temp YAML and call evaluate.main() via subprocess."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="mc_exp_"
    ) as f:
        yaml.dump(cfg, f, default_flow_style=False)
        tmp_path = f.name

    _print_header(cfg)
    print(f"{'='*70}")

    if dry_run:
        print("  [dry-run] skipping")
        os.unlink(tmp_path)
        return 0

    src_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(src_dir, "evaluate.py"),
                "--config",
                tmp_path,
            ],
            check=False,
        )
        return result.returncode
    finally:
        os.unlink(tmp_path)


def run_matrix(matrix_cfg: dict, dry_run: bool, fail_fast: bool):
    defaults = matrix_cfg.get("defaults", {})
    experiments = matrix_cfg.get("experiments", [])

    if not experiments:
        print("No experiments found.")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments.")
    failed = []

    for i, exp in enumerate(experiments):
        cfg = merge(defaults, exp)
        print(f"\n[{i+1}/{len(experiments)}]", end="")
        rc = run_single_subprocess(cfg, dry_run)
        if rc != 0:
            failed.append(_label(cfg))
            if fail_fast:
                print(f"\nFailed (exit {rc}). Stopping.")
                sys.exit(rc)
            print(f"  [WARNING] Failed (exit {rc}), continuing.")

    print(f"\n{'='*70}")
    print(f"Done. {len(experiments) - len(failed)}/{len(experiments)} succeeded.")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")
    print(f"{'='*70}")
    sys.exit(1 if failed else 0)


def run_single(cfg: dict, args):
    """Run a single experiment in-process."""
    from utils import config as cfg_module
    import evaluate

    config_path = args.config
    cfg_module.load_from_yaml(config_path)
    cfg_module.set_seed(cfg_module.seed)
    evaluate.main()


def main():
    parser = argparse.ArgumentParser(description="Run one experiment or a full matrix.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without running."
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure (matrix mode only).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if _is_matrix(cfg):
        run_matrix(cfg, dry_run=args.dry_run, fail_fast=args.fail_fast)
    else:
        _print_header(cfg)
        if args.dry_run:
            print("  [dry-run] skipping")
            return
        run_single(cfg, args)


if __name__ == "__main__":
    main()
