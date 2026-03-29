import json
import random
import torch
from aac_datasets import AudioCaps
from utils import config


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
    # Select one caption per sample and encode it with the fixed tokenizer.
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


def compute_loss(plm: torch.nn.Module, batch: dict) -> torch.Tensor:
    audio = batch["audio"]
    audio_shape = batch["audio_shape"]
    captions = batch["captions"]

    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]

    encoder_outputs = plm.encode_audio(audio, audio_shape)
    logits = plm.decode_audio(
        encoder_outputs,
        "forcing",
        caps_in=captions_in,
    )

    return plm.train_criterion(logits, captions_out)


def finetune_conette(
    hf_model: torch.nn.Module,
    dataset_name: str,
    pruned_layer_names: set[str] = None,
    num_epochs: int = 1,
    lr: float = 5e-6,
    weight_decay: float = 1e-4,
) -> None:
    print(f"Finetuning CoNeTTE on {dataset_name}")

    dataset = AudioCaps(
        config.data_folder,
        subset="train",
        audio_format="wav",
        sr=22050,
        verbose=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    hf_model.train()

    hf_model.requires_grad_(False)
    for name, module in hf_model.named_modules():
        if name in pruned_layer_names:
            module.requires_grad_(True)
            print("Unfroze:", name)

    trainable_parameters = [p for p in hf_model.parameters() if p.requires_grad]
    print("Number of trainable parameter tensors:", len(trainable_parameters))

    trainable_count = sum(p.numel() for p in trainable_parameters)
    total_count = sum(p.numel() for p in hf_model.parameters())
    print(
        f"Trainable params: {trainable_count}/{total_count} ({100 * trainable_count / total_count:.2f}%)"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=lr,
        weight_decay=weight_decay,
    )

    # Sanity check on a single batch before the actual training loop.
    raw_batch = next(iter(train_loader))
    batch = prepare_batch(hf_model, raw_batch, dataset_name)
    loss = compute_loss(hf_model.model, batch)
    print(loss)

    max_steps_per_epoch = 2000

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for step, raw_batch in enumerate(train_loader):
            if step >= max_steps_per_epoch:
                break

            batch = prepare_batch(hf_model, raw_batch, dataset_name)

            optimizer.zero_grad()
            loss = compute_loss(hf_model.model, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if step % 50 == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f}")

        avg_loss = epoch_loss / max_steps_per_epoch
        print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")

        save_dir = f"model/conette_pruned_ac_finetuned_epoch{epoch}"
        hf_model.save_pretrained(save_dir)

        meta = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "lr": lr,
            "dataset": dataset_name,
        }

        with open(f"{save_dir}/meta.json", "x") as file:
            json.dump(meta, file, indent=2)
