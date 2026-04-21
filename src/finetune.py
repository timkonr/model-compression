import random
import torch


def select_captions(raw_captions: list) -> list[str]:
    selected_captions = []

    for captions in raw_captions:
        if isinstance(captions, tuple):
            captions = list(captions)

        if isinstance(captions, list):
            caption = random.choice(captions)
        else:
            caption = captions

        while isinstance(caption, (list, tuple)):
            caption = caption[0]
        selected_captions.append(str(caption))

    return selected_captions


def encode_captions(
    tokenizer, captions: list[str], device: torch.device
) -> torch.Tensor:
    """
    Encodes captions with the fixed CoNeTTE vocabulary.
    Tokens outside the vocabulary are mapped to <unk>.
    """
    tokenized_captions = tokenizer.tokenize_batch(captions)

    bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    unk_token_id = tokenizer.unk_token_id

    encoded_captions = []
    max_length = 0

    for tokens in tokenized_captions:
        tokens = [bos_token] + tokens + [eos_token]

        token_ids = []
        for token in tokens:
            if tokenizer.has(token):
                token_ids.append(tokenizer.token_to_id(token))
            else:
                token_ids.append(unk_token_id)

        encoded_captions.append(token_ids)
        max_length = max(max_length, len(token_ids))

    padded_captions = []
    for token_ids in encoded_captions:
        padded_token_ids = token_ids + [pad_token_id] * (max_length - len(token_ids))
        padded_captions.append(padded_token_ids)

    return torch.tensor(padded_captions, dtype=torch.long, device=device)


def prepare_batch(
    hf_model: torch.nn.Module,
    raw_batch: dict,
    dataset_name: str,
) -> dict:
    # Convert raw audio input to the internal CoNeTTE representation.
    batch = hf_model.preprocessor(raw_batch["audio"], raw_batch["sr"])

    batch = {
        key: value.to(hf_model.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    captions = select_captions(raw_batch["captions"])
    caption_ids = encode_captions(
        hf_model.model.tokenizer,
        captions,
        hf_model.device,
    )

    batch["captions"] = caption_ids

    # Add dataset metadata required by CoNeTTE.
    batch_size = batch["audio"].shape[0]
    batch["dataset"] = [dataset_name] * batch_size
    batch["source"] = [None] * batch_size

    # Replace the generic BOS token with the dataset-specific BOS token.
    batch = hf_model.model.replace_first_ids_in_batch(batch)

    return batch


