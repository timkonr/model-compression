"""Statistical analysis of Raspberry Pi latency measurements.

Compares the baseline with the three distilled, structurally compressed models.
Uses scipy's / statsmodels' tested implementations (paired permutation test,
paired bootstrap CI, Holm correction) instead of hand-rolled ones.

The same audio clips must occur in the same order in every JSON file -- verified
via the "filenames" field saved by measure_latency.py where available.

Usage:
    python src/test_pi_latency_significance.py
"""

import json
import os

import numpy as np
from scipy.stats import bootstrap, permutation_test
from statsmodels.stats.multitest import multipletests


PI_DIR = os.path.join("experiments", "pi_latency")

MODELS = {
    "Enc 70%": "KD_enc_70",
    "Dec 90%": "KD_dec_90",
    "Enc70+Dec90": "KD_enc70_dec90",
}

BASELINE_STEM = "baseline"

N_PERMUTATIONS = 200_000
N_BOOTSTRAPS = 20_000
SEED = 42


def load(stem):
    path = os.path.join(PI_DIR, f"{stem}.json")

    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    if "latencies_ms" not in data:
        raise ValueError(f"{path} does not contain 'latencies_ms'")

    return data


def get_latencies(data):
    latencies = np.asarray(data["latencies_ms"], dtype=float)

    if len(latencies) == 0:
        raise ValueError("Latency list is empty")

    if not np.all(np.isfinite(latencies)):
        raise ValueError("Latency list contains invalid values")

    return latencies


def check_pairing(baseline, model):
    """Check that both files contain the same measurements, in the same order."""

    baseline_latencies = get_latencies(baseline)
    model_latencies = get_latencies(model)

    if len(baseline_latencies) != len(model_latencies):
        raise ValueError(
            "The latency lists have different lengths: "
            f"{len(baseline_latencies)} vs. {len(model_latencies)}"
        )

    # Use clip IDs if they were stored during measurement.
    for key in ("clip_ids", "audio_ids", "filenames"):
        if key in baseline and key in model:
            if baseline[key] != model[key]:
                raise ValueError(f"The samples differ according to '{key}'")
            return True

    return False


def _mean_diff(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def run_permutation_test(baseline, model, rng):
    """Two-sided paired permutation test on the mean difference. Under the null
    hypothesis, baseline and model are exchangeable within each audio clip
    (equivalent to randomly flipping the sign of each paired difference)."""
    result = permutation_test(
        (baseline, model),
        _mean_diff,
        permutation_type="samples",
        vectorized=True,
        n_resamples=N_PERMUTATIONS,
        alternative="two-sided",
        random_state=rng,
    )
    return result.pvalue


def run_bootstrap_ci(baseline, model, rng):
    """Paired (BCa) bootstrap CIs for absolute and relative mean savings."""
    abs_result = bootstrap(
        (baseline, model),
        _mean_diff,
        paired=True,
        vectorized=True,
        n_resamples=N_BOOTSTRAPS,
        confidence_level=0.95,
        random_state=rng,
    )

    def _rel_diff(x, y, axis):
        return 100 * _mean_diff(x, y, axis) / np.mean(x, axis=axis)

    rel_result = bootstrap(
        (baseline, model),
        _rel_diff,
        paired=True,
        vectorized=True,
        n_resamples=N_BOOTSTRAPS,
        confidence_level=0.95,
        random_state=rng,
    )
    absolute_ci = (abs_result.confidence_interval.low, abs_result.confidence_interval.high)
    relative_ci = (rel_result.confidence_interval.low, rel_result.confidence_interval.high)
    return absolute_ci, relative_ci


def holm_correction(p_values):
    """Holm-Bonferroni correction for multiple comparisons (statsmodels)."""
    _, corrected, _, _ = multipletests(p_values, method="holm")
    return corrected


def get_token_counts(data):
    """Return generated-caption word counts if they were stored."""

    for key in (
        "generated_token_counts",
        "output_token_counts",
        "output_lengths",
    ):
        if key in data:
            return np.asarray(data[key], dtype=float)

    return None


def format_p_value(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def main():
    baseline_data = load(BASELINE_STEM)
    baseline = get_latencies(baseline_data)

    results = []
    pairing_verified = True

    for index, (label, stem) in enumerate(MODELS.items()):
        model_data = load(stem)
        model = get_latencies(model_data)

        if not check_pairing(baseline_data, model_data):
            pairing_verified = False

        permutation_rng = np.random.default_rng(SEED + index)
        bootstrap_rng = np.random.default_rng(SEED + 100 + index)

        p_value = run_permutation_test(baseline, model, permutation_rng)
        absolute_ci, relative_ci = run_bootstrap_ci(baseline, model, bootstrap_rng)

        saving_ms = np.mean(baseline) - np.mean(model)
        saving_percent = 100 * saving_ms / np.mean(baseline)

        baseline_tokens = get_token_counts(baseline_data)
        model_tokens = get_token_counts(model_data)

        if baseline_tokens is not None and model_tokens is not None:
            if len(baseline_tokens) != len(model_tokens):
                raise ValueError(f"Token counts differ for {label}")

            token_text = (
                f"{np.mean(baseline_tokens):.1f} -> "
                f"{np.mean(model_tokens):.1f}"
            )
        else:
            token_text = "not stored"

        results.append(
            {
                "label": label,
                "size": model_data.get("model_size_mb"),
                "latency": np.mean(model),
                "saving_ms": saving_ms,
                "absolute_ci": absolute_ci,
                "saving_percent": saving_percent,
                "relative_ci": relative_ci,
                "faster_clips": int(np.sum(model < baseline)),
                "n": len(model),
                "p": p_value,
                "tokens": token_text,
            }
        )

    corrected_p_values = holm_correction([result["p"] for result in results])

    print(f"Baseline mean latency: {np.mean(baseline):.1f} ms")
    print(f"Number of clips: {len(baseline)}")

    if not pairing_verified:
        print(
            "Warning: No clip IDs were found. Pairing therefore relies "
            "on identical dataset order."
        )

    print()
    print(
        "| Model | Size | Mean latency | Saving vs. baseline | "
        "Relative saving | Faster clips | Generated tokens | Holm p |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for result, corrected_p in zip(results, corrected_p_values):
        size = (
            f"{result['size']:.1f} MB"
            if result["size"] is not None
            else "n/a"
        )

        absolute_ci = result["absolute_ci"]
        relative_ci = result["relative_ci"]

        print(
            f"| {result['label']} "
            f"| {size} "
            f"| {result['latency']:.1f} ms "
            f"| {result['saving_ms']:.1f} ms "
            f"[{absolute_ci[0]:.1f}, {absolute_ci[1]:.1f}] "
            f"| {result['saving_percent']:.1f}% "
            f"[{relative_ci[0]:.1f}, {relative_ci[1]:.1f}] "
            f"| {result['faster_clips']}/{result['n']} "
            f"| {result['tokens']} "
            f"| {format_p_value(corrected_p)} |"
        )


if __name__ == "__main__":
    main()
