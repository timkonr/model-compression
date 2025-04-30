import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from conette import CoNeTTEModel, CoNeTTEConfig  # existing imports from your code
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
from transformers import AutoTokenizer, AutoModel
from aac_metrics import evaluate
import config


# Mean pooling helper (unchanged)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / input_mask_expanded.sum(1)


# Distillation loss comparing transformer-based sentence embeddings (unchanged)
def distillation_loss(student_output, teacher_output, tokenizer, transformer, device):
    student_tokens = tokenizer(
        student_output, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    teacher_tokens = tokenizer(
        teacher_output, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    student_out = transformer(**student_tokens)
    teacher_out = transformer(**teacher_tokens)

    student_embed = mean_pooling(student_out, student_tokens["attention_mask"])
    with torch.no_grad():
        teacher_embed = mean_pooling(teacher_out, teacher_tokens["attention_mask"])

    loss = nn.functional.mse_loss(student_embed, teacher_embed)
    return loss


# Validation uses the teacher model for ground-truth outputs
def validate_student(
    student_model, teacher_model, val_loader, tokenizer, transformer, device
):
    student_model.eval()
    teacher_model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["audio"]
            student_output = student_model(inputs)["cands"]
            teacher_output = teacher_model(inputs)["cands"]
            loss = distillation_loss(
                student_output, teacher_output, tokenizer, transformer, device
            )
            total_loss += loss.item()
            count += 1

    return total_loss / count


# Define a custom encoder wrapping EfficientNet-B2
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class EfficientNetB2AudioEncoder(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        # 1) STFT + LogMel from the original ConvNeXt encoder
        self.spectrogram_extractor = original_encoder.spectrogram_extractor
        self.logmel_extractor = original_encoder.logmel_extractor

        # 2) (optional) your augmentation steps
        self.spec_augmenter = original_encoder.spec_augmenter
        self.speed_perturb = original_encoder.speed_perturb

        # 3) Load torchvision’s EfficientNet-B2
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        efb2 = efficientnet_b2(weights=weights)

        # 4) Swap the stem to accept 1 channel
        stem = efb2.features[0][0]
        if stem.in_channels != 1:
            new_stem = nn.Conv2d(
                1,
                stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                bias=False,
            )
            # initialize by averaging the RGB weights
            new_stem.weight.data = stem.weight.data.mean(dim=1, keepdim=True)
            efb2.features[0][0] = new_stem

        # 5) Grab only the final conv features
        return_nodes = {"features.7": "feat"}
        self.extractor = create_feature_extractor(efb2, return_nodes=return_nodes)

        # 6) run a dummy through `self.extractor` at your chosen input size
        with torch.no_grad():
            # pick the exact size you resize to in forward, e.g. (224,224)
            dummy = torch.zeros(1, 1, 224, 224)
            feat = self.extractor(dummy)["feat"]  # [1, C, H', W']
            self.out_channels = feat.shape[1]

    def forward(self, *args, **kwargs):
        # CoNeTTE will call encoder(x, x_shapes).
        # We only care about the waveform x = args[0].
        wave = args[0]  # [B, T]
        # --- STFT & LogMel ---
        x = self.spectrogram_extractor(wave)  # yields [B, 513, time]
        x = self.logmel_extractor(x)  # yields [B, 224, time]

        # --- make it a 1‑channel “image” ---
        x = x.unsqueeze(1)  # [B, 1, 224, time]

        # --- optional audio augmentations ---
        x = self.spec_augmenter(x)
        x = self.speed_perturb(x)
        while x.dim() > 4:
            x = x.squeeze(2)

        # resize to 224x224
        x = nn.functional.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )

        # --- EfficientNet feature extraction ---
        feats = self.extractor(x)["feat"]  # [B, C=1408, H', W']
        B, C, H, W = feats.shape

        # flatten spatial → tokens
        out = feats.view(B, C, H * W).transpose(1, 2)  # [B, tokens, C]
        out = nn.functional.layer_norm(out, (out.size(-1),), eps=1e-6)

        batch_size, seq_len, _ = out.size()
        frame_embs_lens = torch.full(
            (batch_size,), seq_len, dtype=torch.long, device=out.device
        )
        return {"frame_embs": out, "frame_embs_lens": frame_embs_lens}


class Projection(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.5):
        super().__init__()
        self.dropout1 = nn.Dropout(p)
        self.lin = nn.Linear(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p)

    def forward(self, x):
        # x: [B, C, T]
        x = self.dropout1(x)
        x = x.transpose(1, 2)  # → [B, T, C]
        x = self.lin(x)  # → [B, T, out_ch]
        x = self.relu(x)
        x = x.transpose(1, 2)  # → [B, out_ch, T]
        return self.dropout2(x)


# Main training function incorporating the student model with EfficientNet-B2 as the encoder
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = config.model_folder + "baseline/"
    print("Loading teacher")
    teacher_config = CoNeTTEConfig.from_pretrained(model_path)
    teacher_model = CoNeTTEModel.from_pretrained(model_path, config=teacher_config)
    teacher_model.to(device).eval()

    print("Defining student")
    # Instantiate student model from teacher config.
    student_config = CoNeTTEConfig.from_pretrained(model_path)
    student_model = CoNeTTEModel(student_config)

    student_model.preprocessor.encoder = EfficientNetB2AudioEncoder(
        original_encoder=teacher_model.preprocessor.encoder
    )

    enc_ch = student_model.preprocessor.encoder.out_channels
    dec_ch = student_config.d_model  # e.g. 256
    student_model.model.projection = Projection(enc_ch, dec_ch, p=0.5)

    # Transfer the tokenizer for consistent text processing.
    student_model.model.tokenizers["0"] = teacher_model.model.tokenizers["0"]
    student_model = student_model.to(device)

    print("Preparing data")
    train_dataset = Clotho(config.data_folder, subset="dev")
    val_dataset = Clotho(config.data_folder, subset="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        collate_fn=BasicCollate(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        collate_fn=BasicCollate(),
    )

    optimizer = optim.AdamW(student_model.parameters(), lr=1e-4)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    transformer = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    ).to(device)

    print("Starting knowledge distillation training")
    best_val_loss = float("inf")
    num_epochs = 20

    for epoch in range(num_epochs):
        student_model.train()
        total_loss = 0.0
        for batch in train_loader:
            inputs = batch["audio"]
            with torch.no_grad():
                teacher_output = teacher_model(inputs)["cands"]
            student_output = student_model(inputs)["cands"]

            loss = distillation_loss(
                student_output, teacher_output, tokenizer, transformer, device
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_train_loss:.4f}")
        val_loss = validate_student(
            student_model, teacher_model, val_loader, tokenizer, transformer, device
        )
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                student_model.state_dict(),
                config.model_folder + "best_student_model.pth",
            )
            print("Saved best student model.")


if __name__ == "__main__":
    main()
