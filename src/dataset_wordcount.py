from aac_datasets import Clotho, AudioCaps
from utils import config
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])


def count_words(text: str):
    doc = nlp(text.replace(".", "").replace(",", ""))
    return len(doc)


def count_words_in_dataset(dataset_name):
    if dataset_name == "clotho":
        ds = Clotho(config.data_folder, subset="eval")
    elif dataset_name == "audiocaps":
        ds = AudioCaps(config.data_folder, subset="test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    total_words = 0
    min_words = float("inf")
    max_words = 0
    min_words_caption = ""
    max_words_caption = ""
    total_captions = 0
    for i, batch in enumerate(ds):
        captions = batch["captions"]
        total_captions += len(captions)
        for caption in captions:
            caption_length = count_words(caption)
            total_words += caption_length
            min_words = min(min_words, caption_length)
            max_words = max(max_words, caption_length)
            if caption_length == min_words:
                min_words_caption = caption
            if caption_length == max_words:
                max_words_caption = caption
    avg_words = total_words / total_captions if total_captions > 0 else 0

    print(f"Total words in {dataset_name} dataset: {total_words}")
    print(f"Minimum words in {dataset_name} dataset: {min_words}")
    print(f"Maximum words in {dataset_name} dataset: {max_words}")
    print(f"Average words in {dataset_name} dataset: {avg_words}")
    print(f"Caption with minimum words in {dataset_name} dataset: {min_words_caption}")
    print(f"Caption with maximum words in {dataset_name} dataset: {max_words_caption}")


def main():
    count_words_in_dataset("clotho")
    count_words_in_dataset("audiocaps")


if __name__ == "__main__":
    main()
