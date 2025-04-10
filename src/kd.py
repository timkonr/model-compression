import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from conette import CoNeTTEModel, CoNeTTEConfig  # existing imports
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
    # Tokenize the input sentences
    student_tokens = tokenizer(
        student_output, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    teacher_tokens = tokenizer(
        teacher_output, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # Get embeddings from the transformer
    student_out = transformer(**student_tokens)
    teacher_out = transformer(**teacher_tokens)

    # Mean pooling to get sentence embeddings
    student_embed = mean_pooling(student_out, student_tokens["attention_mask"])
    with torch.no_grad():
        teacher_embed = mean_pooling(teacher_out, teacher_tokens["attention_mask"])

    # MSE loss between embeddings
    loss = nn.functional.mse_loss(student_embed, teacher_embed)
    return loss


# Change validation to use teacher_model for ground-truth outputs
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


# Define a custom ConvNeXt block with a compatible signature
class SmallConvNeXtBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
        )
        self.norm = nn.LayerNorm(in_channels)
        self.pwconv1 = nn.Linear(in_channels, in_channels * 4)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(in_channels * 4, in_channels)
        self.drop_path = nn.Identity()  # Replace with DropPath if needed

    def forward(self, x):
        # x: [B, C, H, W]
        residual = x
        x = self.dwconv(x)  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # Back to [B, C, H, W]
        x = self.drop_path(x)
        return x + residual


# Updated custom encoder that accepts extra arguments (if any)
class SmallConvNeXtEncoder(nn.Module):
    def __init__(self, original_encoder):
        """
        Construct a slimmed-down encoder.
        This example reinitializes the downsampling layers and stages with lower widths and fewer blocks.
        """
        super().__init__()
        # Retain preprocessing parts from original encoder
        self.spectrogram_extractor = original_encoder.spectrogram_extractor
        self.logmel_extractor = original_encoder.logmel_extractor
        self.spec_augmenter = original_encoder.spec_augmenter
        self.speed_perturb = original_encoder.speed_perturb
        self.bn0 = original_encoder.bn0

        # Build new downsampling layers with smaller output channels
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0)),
                    nn.LayerNorm([64, 1, 1]),  # Adjust shape if needed
                ),
                nn.Sequential(
                    nn.LayerNorm(64),
                    nn.Conv2d(64, 128, kernel_size=(2, 2), stride=(2, 2)),
                ),
                nn.Sequential(
                    nn.LayerNorm(128),
                    nn.Conv2d(128, 256, kernel_size=(2, 2), stride=(2, 2)),
                ),
                nn.Sequential(
                    nn.LayerNorm(256),
                    nn.Conv2d(256, 512, kernel_size=(2, 2), stride=(2, 2)),
                ),
            ]
        )

        # Build stages with our custom small block.
        # Proposed depths: [2, 2, 6, 2] for the four stages.
        stage_depths = [2, 2, 6, 2]
        stage_channels = [64, 128, 256, 512]
        self.stages = nn.ModuleList()
        for channels, depth in zip(stage_channels, stage_depths):
            blocks = []
            for _ in range(depth):
                blocks.append(SmallConvNeXtBlock(in_channels=channels))
            self.stages.append(nn.Sequential(*blocks))

        # Final normalization layer for output features
        self.norm = nn.LayerNorm(512)  # final channels = 512

    def forward(self, *args, **kwargs):
        """
        Accept extra positional and keyword arguments so that the call signature
        matches the original encoder. We'll only use the first positional argument as input.
        """
        x = args[0]  # assume the first positional argument is the input tensor

        # Preprocessing
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = self.spec_augmenter(x)
        x = self.speed_perturb(x)
        x = self.bn0(x)

        # Downsampling layers
        for layer in self.downsample_layers:
            x = layer(x)

        # ConvNeXt stages
        for stage in self.stages:
            x = stage(x)

        # Flatten spatial dimensions: [B, C, H, W] -> [B, tokens, C]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)
        x = self.norm(x)
        return x


# Custom projection layer: maps encoder output (512) to decoder expected dimension (256)
class CustomProjection(nn.Module):
    def __init__(self, in_dim=512, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.proj(x))


# Main training function incorporating the student model with the custom smaller encoder
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/"
    print("Loading teacher")
    teacher_config = CoNeTTEConfig.from_pretrained(model_path)
    teacher_model = CoNeTTEModel.from_pretrained(model_path, config=teacher_config)
    teacher_model.to(device).eval()

    print("Defining student")
    # Instantiate student model based on teacher config
    student_config = CoNeTTEConfig.from_pretrained(model_path)
    student_model = CoNeTTEModel(student_config)

    # Replace the encoder with our custom smaller encoder
    student_model.preprocessor.encoder = SmallConvNeXtEncoder(
        teacher_model.preprocessor.encoder
    )

    # Replace the projection layer to map from 512 (new encoder output) to d_model (256)
    student_model.model.projection = nn.Sequential(
        nn.Dropout(p=0.5),
        CustomProjection(in_dim=512, out_dim=student_config.d_model),
        nn.ReLU(),
        nn.Dropout(p=0.5),
    )

    # Transfer the tokenizer (ensuring vocabulary alignment)
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
            torch.save(student_model.state_dict(), "best_student_model.pth")
            print("Saved best student model.")


if __name__ == "__main__":
    main()
