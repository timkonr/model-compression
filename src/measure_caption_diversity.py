import random
from statistics import mean, stdev

from aac_datasets import Clotho, AudioCaps
from utils import config


def flatten_captions(ds):
    caps = []
    for item in ds:
        caps.extend(item["captions"])
    return caps


def stats(name, captions):
    tokenized = [c.split() for c in captions]
    vocab = set(t for toks in tokenized for t in toks)
    lengths = [len(toks) for toks in tokenized]
    n = len(captions)
    n_tokens = sum(lengths)
    result = {
        "n_captions": n,
        "vocab_size": len(vocab),
        "type_token_ratio": len(vocab) / n_tokens,
        "mean_caption_len": n_tokens / n,
        "unique_caption_frac": len(set(captions)) / n,
    }
    print(f"{name}:")
    print(f"  captions:             {result['n_captions']}")
    print(f"  vocabulary size:      {result['vocab_size']}")
    print(f"  type-token ratio:     {result['type_token_ratio']:.4f}")
    print(f"  mean caption length:  {result['mean_caption_len']:.2f} words")
    print(f"  unique caption frac.: {result['unique_caption_frac']:.4f}")
    print()
    return result


def subsample_ttr(captions, target_tokens, n_iter=200, seed=0):
    """Repeatedly draw random whole captions (shuffled order, respecting caption
    boundaries) until the token budget is reached, controlling for the corpus-size
    confound in TTR (TTR shrinks mechanically as token count grows, independent of
    genuine diversity)."""
    rng = random.Random(seed)
    tokenized = [c.split() for c in captions]
    vocab_sizes, ttrs = [], []
    for _ in range(n_iter):
        order = list(range(len(tokenized)))
        rng.shuffle(order)
        vocab = set()
        n_tokens = 0
        for idx in order:
            toks = tokenized[idx]
            vocab.update(toks)
            n_tokens += len(toks)
            if n_tokens >= target_tokens:
                break
        vocab_sizes.append(len(vocab))
        ttrs.append(len(vocab) / n_tokens)
    return vocab_sizes, ttrs


def main():
    print("loading Clotho dev (calibration pool)...")
    clotho_ds = Clotho(config.data_folder, subset="dev")
    clotho_caps = flatten_captions(clotho_ds)

    print("loading AudioCaps val (calibration pool)...")
    ac_ds = AudioCaps(config.data_folder, subset="val", audio_format="wav", sr=22050)
    ac_caps = flatten_captions(ac_ds)

    stats("Clotho dev", clotho_caps)
    ac_result = stats("AudioCaps val", ac_caps)

    ac_n_tokens = round(ac_result["mean_caption_len"] * ac_result["n_captions"])
    print(
        f"subsampling Clotho dev down to AudioCaps' token budget "
        f"(~{ac_n_tokens} tokens, 200 draws)..."
    )
    vocab_sizes, ttrs = subsample_ttr(clotho_caps, ac_n_tokens)
    v_mean, v_sd = mean(vocab_sizes), stdev(vocab_sizes)
    t_mean, t_sd = mean(ttrs), stdev(ttrs)
    print("Clotho dev, subsampled to AudioCaps' token budget:")
    print(f"  vocabulary size: {v_mean:.1f} +/- {v_sd:.1f}")
    print(f"  type-token ratio: {t_mean:.4f} +/- {t_sd:.4f}")
    print()

    ac_ttr = ac_result["type_token_ratio"]
    z = (ac_ttr - t_mean) / t_sd if t_sd > 0 else float("nan")
    verdict = "AudioCaps HIGHER" if ac_ttr > t_mean else "Clotho HIGHER"
    print(
        f"AudioCaps val TTR ({ac_ttr:.4f}) vs. size-matched Clotho dev TTR "
        f"({t_mean:.4f} +/- {t_sd:.4f}) -> z={z:+.2f} -> {verdict} "
        f"(at matched token budget, corpus-size confound removed)"
    )


if __name__ == "__main__":
    main()
