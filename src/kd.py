import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from conette import CoNeTTEModel, CoNeTTEConfig
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import AutoTokenizer, AutoModel
import config
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def pad_audio(audio_list, device):
    """
    audio_list: list of 1D or 2D tensors (e.g. [T] or [1, T]).
    Returns:
      padded: Tensor of shape [B, T_max] on `device`
      lengths: list of original lengths
    """
    processed = []
    for a in audio_list:
        # collapse any singleton channel dimension
        if a.dim() == 2 and a.size(0) == 1:
            a = a.squeeze(0)  # [1, T] -> [T]
        processed.append(a)

    lengths = [a.size(-1) for a in processed]
    max_len = max(lengths)

    padded = torch.stack(
        [F.pad(a, (0, max_len - L)) for a, L in zip(processed, lengths)], dim=0
    )  # [B, T_max]

    return padded.to(device, non_blocking=True), lengths


def extract_proj(model, audio_list, device):
    # 1) Pad & batch → [B, T] on the correct device + lengths
    inputs, lengths = pad_audio(audio_list, device)

    # 2) Build x_shapes tensor ([B,2]) so ConvNeXt can index it
    lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
    x_shapes = torch.stack(
        [
            torch.zeros_like(lengths_t),  # channel dimension placeholder
            lengths_t,  # time dimension
        ],
        dim=1,
    )  # shape [B,2]

    # 3) Encoder → {"frame_embs": [B,T,C], ...}
    outs = model.preprocessor.encoder(inputs, x_shapes)
    fe = outs["frame_embs"].transpose(1, 2).contiguous()  # [B,C,T]

    # 4) Projection → [B,d_model,T]
    return model.model.projection(fe)


# ------------------------------------------------------------------
# Custom EfficientNet-B2 Encoder
# ------------------------------------------------------------------
class EfficientNetB2AudioEncoder(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        # 1) STFT & LogMel preprocessor
        self.spectrogram_extractor = original_encoder.spectrogram_extractor
        self.logmel_extractor = original_encoder.logmel_extractor
        self.spec_augmenter = original_encoder.spec_augmenter
        self.speed_perturb = original_encoder.speed_perturb

        # 2) Load EfficientNet-B2 & swap to 1-channel stem
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        efb2 = efficientnet_b2(weights=weights)
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
            new_stem.weight.data = stem.weight.data.mean(dim=1, keepdim=True)
            efb2.features[0][0] = new_stem

        return_nodes = {"features.7": "feat"}
        self.extractor = create_feature_extractor(efb2, return_nodes=return_nodes)

        # 3) determine out_channels via dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224)
            self.out_channels = self.extractor(dummy)["feat"].shape[1]

    def forward(self, audio, lengths):
        """
        audio: [B, T] on GPU
        lengths: list of original lengths
        """
        # STFT → LogMel
        x = self.spectrogram_extractor(audio)
        x = self.logmel_extractor(x)  # [B, 224, time]
        x = x.unsqueeze(1)  # [B,1,224,time]
        x = self.spec_augmenter(x)
        x = self.speed_perturb(x)
        while x.dim() > 4:
            x = x.squeeze(2)

        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)

        # EfficientNet features
        feats = self.extractor(x)["feat"]  # [B, C, H', W']
        B, C, H, W = feats.shape

        # flatten → [B, T, C]
        out = feats.view(B, C, H * W).transpose(1, 2)
        out = F.layer_norm(out, (out.size(-1),), eps=1e-6)

        # lengths for masking
        lens = torch.full((B,), out.size(1), dtype=torch.long, device=out.device)
        return {"frame_embs": out, "frame_embs_lens": lens}


# ------------------------------------------------------------------
# Simple Projection
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Validation (feature-level distillation)
# ------------------------------------------------------------------
def validate_student(student, teacher, loader, device):
    student.eval()
    teacher.eval()
    total_loss, n = 0.0, 0
    val_bar = tqdm(loader, desc=f"Validation", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            audios = batch["audio"]  # list of T_i
            t_proj = extract_proj(teacher, audios, device)
            s_proj = extract_proj(student, audios, device)
            if t_proj.size(2) != s_proj.size(2):
                t_proj = F.adaptive_avg_pool1d(t_proj, s_proj.size(2))
            total_loss += F.mse_loss(s_proj, t_proj).item()
            n += 1

    return total_loss / n


# ------------------------------------------------------------------
# Main training
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Teacher
    base_path = config.model_folder + "baseline/"
    teacher_cfg = CoNeTTEConfig.from_pretrained(base_path)
    teacher_model = CoNeTTEModel.from_pretrained(base_path, config=teacher_cfg)
    teacher_model.to(device).eval()

    # Student
    student_cfg = CoNeTTEConfig.from_pretrained(base_path)
    student_model = CoNeTTEModel(student_cfg)
    student_model.preprocessor.encoder = EfficientNetB2AudioEncoder(
        teacher_model.preprocessor.encoder
    )
    enc_ch = student_model.preprocessor.encoder.out_channels
    dec_ch = student_cfg.d_model
    student_model.model.projection = Projection(enc_ch, dec_ch, p=0.5)
    student_model.model.tokenizers["0"] = teacher_model.model.tokenizers["0"]
    student_model.to(device)

    # Data
    train_loader = DataLoader(
        Clotho(config.data_folder, subset="dev"),
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=BasicCollate(),
    )
    val_loader = DataLoader(
        Clotho(config.data_folder, subset="val"),
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=BasicCollate(),
    )

    optimizer = optim.AdamW(student_model.parameters(), lr=1e-4)

    print("Starting distillation…")
    best_val = float("inf")
    num_epochs = 20

    for epoch in range(1, num_epochs + 1):
        student_model.train()
        total_loss = 0.0

        train_bar = tqdm(
            train_loader, desc=f"[Epoch {epoch}/{num_epochs}] Training", leave=False
        )
        scaler = GradScaler()
        for batch in train_bar:
            audios = batch["audio"]
            with torch.no_grad():
                t_proj = extract_proj(teacher_model, audios, device)

            optimizer.zero_grad()
            with autocast():
                s_proj = extract_proj(student_model, audios, device)

                if t_proj.size(2) != s_proj.size(2):
                    t_proj = F.adaptive_avg_pool1d(t_proj, s_proj.size(2))
                loss = F.mse_loss(s_proj, t_proj)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        val_loss = validate_student(student_model, teacher_model, val_loader, device)
        print(f"[Epoch {epoch}] train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                student_model.state_dict(),
                config.model_folder + config.kd_model,
            )
            print("  → Saved new best")


if __name__ == "__main__":
    main()
