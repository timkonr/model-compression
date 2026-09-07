import argparse
import random
from collections import Counter
from itertools import chain, combinations
from pathlib import Path
from statistics import mean, stdev

import torch
from aac_datasets import AudioCaps, Clotho
from conette import CoNeTTEConfig, CoNeTTEModel

from utils import config


DEFAULT_SEEDS = (42, 43, 44)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure the lexical diversity of the calibration samples actually "
            "used for activation-aware pruning."
        )
    )
    parser.add_argument("--n-clips", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--mattr-window", type=int, default=100)
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the seeded shuffled clip selection from the repeated calibration "
            "experiment. Use --no-shuffle for the earlier deterministic clip draw."
        ),
    )
    parser.add_argument("--data-folder", default=config.data_folder)
    parser.add_argument("--model-folder", default=config.model_folder)
    return parser.parse_args()


def _flatten_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_strings(item)
    else:
        yield str(value)


def load_caption_pool(dataset):
    """Return one list of reference captions per clip, preserving dataset order."""
    pool = []
    for idx in range(len(dataset)):
        references = list(_flatten_strings(dataset[idx]["captions"]))
        if not references:
            raise RuntimeError(f"Clip {idx} does not contain a caption.")
        pool.append(references)
    return pool


def load_conette_tokenizer(model_folder):
    baseline_path = Path(model_folder) / "baseline"
    model = CoNeTTEModel.from_pretrained(
        str(baseline_path),
        config=CoNeTTEConfig.from_pretrained(str(baseline_path)),
    )
    tokenizer = model.model.tokenizer
    del model
    return tokenizer


def tokenize_captions(tokenizer, captions, batch_size=512):
    tokenized = []
    for start in range(0, len(captions), batch_size):
        batch = captions[start : start + batch_size]
        tokenized.extend([list(tokens) for tokens in tokenizer.tokenize_batch(batch)])
    return tokenized


def moving_average_ttr(tokens, window):
    if not tokens:
        return float("nan")
    if window <= 0:
        raise ValueError("MATTR window must be positive.")
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return mean(
        len(set(tokens[start : start + window])) / window
        for start in range(len(tokens) - window + 1)
    )


def compute_stats(tokenizer, captions, pool_token_counts, mattr_window):
    tokenized = tokenize_captions(tokenizer, captions)
    tokens = list(chain.from_iterable(tokenized))
    vocabulary = set(tokens)
    n_tokens = len(tokens)

    if not captions or not tokens:
        raise RuntimeError("Cannot compute diversity statistics for an empty sample.")

    unknown_tokens = sum(not tokenizer.has(token) for token in tokens)
    represented_pool_tokens = sum(
        count for token, count in pool_token_counts.items() if token in vocabulary
    )

    return {
        "captions": len(captions),
        "tokens": n_tokens,
        "types": len(vocabulary),
        "ttr": len(vocabulary) / n_tokens,
        "mattr": moving_average_ttr(tokens, mattr_window),
        "mean_length": mean(map(len, tokenized)),
        "unique_fraction": len({tuple(tokens) for tokens in tokenized}) / len(tokenized),
        "unk_rate": unknown_tokens / n_tokens,
        "pool_token_coverage": represented_pool_tokens / sum(pool_token_counts.values()),
        "vocabulary": vocabulary,
    }


def calibration_draw(caption_pool, n_clips, seed, shuffle):
    """Mirror evaluate.prepare_dataloader and finetune.select_captions."""
    if n_clips > len(caption_pool):
        raise ValueError(
            f"Requested {n_clips} clips from a pool containing {len(caption_pool)}."
        )

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(caption_pool), generator=generator)[:n_clips].tolist()
    else:
        indices = list(range(n_clips))

    caption_rng = random.Random(seed)
    captions = [caption_rng.choice(caption_pool[idx]) for idx in indices]
    return indices, captions


def pairwise_jaccard(sets):
    values = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return mean(values) if values else float("nan")


def mean_sd(values):
    if len(values) == 1:
        return mean(values), 0.0
    return mean(values), stdev(values)


def print_pool_summary(name, caption_pool, pool_stats, n_clips, mattr_window):
    print(f"{name} calibration pool")
    print(f"  clips:                  {len(caption_pool)}")
    print(f"  reference captions:     {pool_stats['captions']}")
    print(f"  tokens:                 {pool_stats['tokens']}")
    print(f"  vocabulary size:        {pool_stats['types']}")
    print(f"  TTR:                    {pool_stats['ttr']:.4f}")
    print(f"  MATTR-{mattr_window}:              {pool_stats['mattr']:.4f}")
    print(f"  mean caption length:    {pool_stats['mean_length']:.2f} tokens")
    print(f"  unique caption frac.:   {pool_stats['unique_fraction']:.4f}")
    print(f"  tokenizer <unk> rate:   {pool_stats['unk_rate']:.4f}")
    print(f"  clips sampled per run:  {n_clips / len(caption_pool):.2%}")
    print()


def print_sample_results(name, results, mattr_window):
    print(f"{name} actual calibration draws")
    print(
        f"  {'seed':>4} {'tokens':>7} {'types':>6} {'TTR':>7} "
        f"{'MATTR-' + str(mattr_window):>10} {'unique':>8} {'<unk>':>8} "
        f"{'pool coverage':>13}"
    )
    for result in results:
        stats = result["stats"]
        print(
            f"  {result['seed']:>4} {stats['tokens']:>7} {stats['types']:>6} "
            f"{stats['ttr']:>7.4f} {stats['mattr']:>10.4f} "
            f"{stats['unique_fraction']:>8.4f} {stats['unk_rate']:>8.4f} "
            f"{stats['pool_token_coverage']:>13.4f}"
        )

    print("  across seeds")
    for key, label in (
        ("tokens", "tokens"),
        ("types", "types"),
        ("ttr", "TTR"),
        ("mattr", f"MATTR-{mattr_window}"),
        ("unique_fraction", "unique fraction"),
        ("unk_rate", "<unk> rate"),
        ("pool_token_coverage", "pool token coverage"),
    ):
        avg, sd = mean_sd([result["stats"][key] for result in results])
        precision = 1 if key in {"tokens", "types"} else 4
        print(f"    {label:<20} {avg:.{precision}f} +/- {sd:.{precision}f}")

    print(
        "    pairwise clip Jaccard "
        f"{pairwise_jaccard([set(result['indices']) for result in results]):.4f}"
    )
    print(
        "    pairwise vocab Jaccard "
        f"{pairwise_jaccard([result['stats']['vocabulary'] for result in results]):.4f}"
    )
    print()


def analyse_dataset(
    name,
    caption_pool,
    tokenizer,
    seeds,
    n_clips,
    shuffle,
    mattr_window,
):
    all_captions = list(chain.from_iterable(caption_pool))
    pool_tokenized = tokenize_captions(tokenizer, all_captions)
    pool_token_counts = Counter(chain.from_iterable(pool_tokenized))
    pool_stats = compute_stats(
        tokenizer,
        all_captions,
        pool_token_counts=pool_token_counts,
        mattr_window=mattr_window,
    )

    results = []
    for seed in seeds:
        indices, captions = calibration_draw(caption_pool, n_clips, seed, shuffle)
        sample_stats = compute_stats(
            tokenizer,
            captions,
            pool_token_counts=pool_token_counts,
            mattr_window=mattr_window,
        )
        results.append({"seed": seed, "indices": indices, "stats": sample_stats})

    print_pool_summary(name, caption_pool, pool_stats, n_clips, mattr_window)
    print_sample_results(name, results, mattr_window)
    return results


def main():
    args = parse_args()
    print("Loading CoNeTTE tokenizer...")
    tokenizer = load_conette_tokenizer(args.model_folder)

    print("Loading Clotho dev caption pool...")
    clotho_pool = load_caption_pool(Clotho(args.data_folder, subset="dev"))
    print("Loading AudioCaps val caption pool...")
    audiocaps_pool = load_caption_pool(
        AudioCaps(
            args.data_folder,
            subset="val",
            audio_format="wav",
            sr=22050,
        )
    )
    print()
    print("Calibration design")
    print(f"  clips per draw:         {args.n_clips}")
    print(f"  seeds:                  {', '.join(map(str, args.seeds))}")
    print(f"  shuffled clip draw:     {args.shuffle}")
    print("  captions per clip:      one random reference")
    print("  tokenization:           CoNeTTE tokenizer")
    print()

    analyse_dataset(
        "Clotho dev",
        clotho_pool,
        tokenizer,
        args.seeds,
        args.n_clips,
        args.shuffle,
        args.mattr_window,
    )
    analyse_dataset(
        "AudioCaps val",
        audiocaps_pool,
        tokenizer,
        args.seeds,
        args.n_clips,
        args.shuffle,
        args.mattr_window,
    )

    print("Interpretation note")
    print(
        "  Prefer MATTR over raw TTR for the cross-dataset comparison because the "
        "caption lengths and therefore the sample token counts differ. Pool token "
        "coverage and pairwise vocabulary Jaccard describe representativeness and "
        "seed sensitivity; they are descriptive statistics, not significance tests."
    )


if __name__ == "__main__":
    main()
