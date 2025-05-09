import torch
import torch.nn as nn
import torch.nn.functional as F

from conette import CoNeTTEModel, CoNeTTEConfig
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import config


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

        out = feats.view(B, C, H * W)  # [B, C, T]   <- no .transpose
        out = F.layer_norm(out.transpose(1, 2), (C,), eps=1e-6)  # LN on C dim
        # transpose back to [B, C, T] if you LN this way, or LN per token
        lens = torch.full((B,), out.size(2), dtype=torch.long, device=out.device)
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


def load_student_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the saved config + create model skeleton
    base_path = config.model_folder + "baseline/"
    student_cfg = CoNeTTEConfig.from_pretrained(base_path)
    student_model = CoNeTTEModel(student_cfg)

    teacher_cfg = CoNeTTEConfig.from_pretrained(base_path)
    teacher_model = CoNeTTEModel.from_pretrained(base_path, config=teacher_cfg).to(
        device
    )
    orig_enc = teacher_model.preprocessor.encoder

    student_model.preprocessor.encoder = EfficientNetB2AudioEncoder(orig_enc)
    enc_ch = student_model.preprocessor.encoder.out_channels
    dec_ch = student_cfg.d_model
    student_model.model.projection = Projection(enc_ch, dec_ch, p=0.5)

    # copy tokenizer for text decoding
    student_model.model.tokenizers["0"] = teacher_model.model.tokenizers["0"]

    # load weights
    checkpoint = torch.load(config.model_folder + config.kd_model, map_location=device)
    student_model.load_state_dict(checkpoint)

    # move & eval
    student_model.to(device)
    student_model.eval()

    return student_model
