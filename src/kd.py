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


class EfficientNetB2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) Load torchvision’s EfficientNet‑B2 (pretrained)
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        base_model = efficientnet_b2(weights=weights)

        # 2) Replace the stem conv to accept 1‑channel input
        stem_conv = base_model.features[0][0]  # Conv2d layer
        if stem_conv.in_channels != 1:
            new_stem = nn.Conv2d(
                1,
                stem_conv.out_channels,
                kernel_size=stem_conv.kernel_size,
                stride=stem_conv.stride,
                padding=stem_conv.padding,
                bias=False,
            )
            # initialize by averaging pretrained weights over input channels
            new_stem.weight.data = stem_conv.weight.data.mean(dim=1, keepdim=True)
            base_model.features[0][0] = new_stem

        # 3) Use feature_extraction to grab the last conv feature map
        #    In EfficientNet‑B2, features[7] is the final block
        return_nodes = {"features.7": "feat"}
        self.extractor = create_feature_extractor(base_model, return_nodes=return_nodes)

        # 4) Get the channel count by inspecting the classifier’s in_features
        #    classifier = [Dropout, Linear(in_features=1408, out_features=1000)]
        self.out_channels = base_model.classifier[1].in_features

        # 5) LayerNorm over that channel dimension
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, *args, **kwargs):
        # Accept extra args/kwargs; use the first positional as input
        x = args[0]  # [B, 1, H, W] spectrogram
        feats = self.extractor(x)["feat"]  # [B, C, H', W']
        B, C, H, W = feats.shape
        # Flatten spatial dims to a token sequence
        out = feats.view(B, C, H * W).transpose(1, 2)  # [B, tokens, C]
        return self.norm(out)


# Custom projection layer: maps the encoder output dimension (from EfficientNet-B2) to the decoder's embedding size (e.g., 256)
class CustomProjection(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.proj(x))


# Main training function incorporating the student model with EfficientNet-B2 as the encoder
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/baseline/"
    print("Loading teacher")
    teacher_config = CoNeTTEConfig.from_pretrained(model_path)
    teacher_model = CoNeTTEModel.from_pretrained(model_path, config=teacher_config)
    teacher_model.to(device).eval()

    print("Defining student")
    # Instantiate student model from teacher config.
    student_config = CoNeTTEConfig.from_pretrained(model_path)
    student_model = CoNeTTEModel(student_config)

    # Replace the encoder with EfficientNet-B2 encoder.
    student_model.preprocessor.encoder = EfficientNetB2Encoder()

    # Replace the projection layer.
    # EfficientNet-B2's last feature dimension is stored in student_model.preprocessor.encoder.out_channels.
    student_model.model.projection = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(
            student_model.preprocessor.encoder.out_channels, student_config.d_model
        ),
        nn.ReLU(),
        nn.Dropout(0.5),
    )

    # Transfer the tokenizer for consistent text processing.
    student_model.model.tokenizers["0"] = teacher_model.model.tokenizers["0"]

    print("Preparing data")
    train_dataset = Clotho("data", subset="dev")
    val_dataset = Clotho("data", subset="val")
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
            torch.save(student_model.state_dict(), "model/best_student_model.pth")
            print("Saved best student model.")


if __name__ == "__main__":
    main()
