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


def main():
    print("loading Clotho dev (calibration pool)...")
    clotho_ds = Clotho(config.data_folder, subset="dev")
    clotho_caps = flatten_captions(clotho_ds)

    print("loading AudioCaps val (calibration pool)...")
    ac_ds = AudioCaps(config.data_folder, subset="val", audio_format="wav", sr=22050)
    ac_caps = flatten_captions(ac_ds)

    stats("Clotho dev", clotho_caps)
    stats("AudioCaps val", ac_caps)


if __name__ == "__main__":
    main()
