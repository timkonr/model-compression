import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from conette import CoNeTTEModel, CoNeTTEConfig
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
from transformers import AutoTokenizer, AutoModel
from aac_metrics import evaluate


# Mean pooling
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


# Example custom smaller ConvNeXt encoder
class SmallConvNeXtEncoder(nn.Module):
    def __init__(self, original_encoder):
        """
        Create a slimmed-down encoder based on the original ConvNeXt encoder.
        This example assumes you can re-use the same building blocks but modify:
          - The number of channels per stage
          - The number of blocks per stage
        """
        super().__init__()
        # You likely have a preprocessor that includes:
        # spectrogram_extractor, logmel_extractor, etc.
        # We extract relevant parts from the original encoder:
        self.spectrogram_extractor = original_encoder.spectrogram_extractor
        self.logmel_extractor = original_encoder.logmel_extractor
        self.spec_augmenter = original_encoder.spec_augmenter
        self.speed_perturb = original_encoder.speed_perturb
        self.bn0 = original_encoder.bn0

        # Downsampling layers can be reused if dimensions still match:
        # (We assume these layers produce a slightly different channel count now)
        # Option 1: Adapt these as well. Option 2: reinitialize them.
        # Here we reinitialize for simplicity.
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0)),
                    nn.LayerNorm(
                        [64, 1, 1]
                    ),  # dummy shape for LayerNorm; adjust accordingly
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

        # Rebuild the stages with fewer ConvNeXt blocks:
        # For example, teacher stages might be: [3, 3, 9, 3]
        # We propose: [2, 2, 6, 2]
        # We assume there is a ConvNeXtBlock class available
        ConvNeXtBlock = original_encoder.stages[0][0].__class__  # infer block class

        # Define each stage
        self.stages = nn.ModuleList()
        stage_dims = [64, 128, 256, 512]
        stage_depths = [2, 2, 6, 2]
        for stage_idx, depth in enumerate(stage_depths):
            blocks = []
            # For each block, adjust the in/out dimensions accordingly:
            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        # You need to check what arguments ConvNeXtBlock requires.
                        # Here we assume it takes: (in_channels, expansion_channels).
                        # We use stage_dims[stage_idx] and multiply by 4 for expansion.
                        # In the teacher, these were (e.g.) 96->384; now, 64->256.
                        dwconv=nn.Conv2d(
                            stage_dims[stage_idx],
                            stage_dims[stage_idx],
                            kernel_size=7,
                            stride=1,
                            padding=3,
                            groups=stage_dims[stage_idx],
                        ),
                        norm=nn.LayerNorm(stage_dims[stage_idx]),
                        pwconv1=nn.Linear(
                            stage_dims[stage_idx], stage_dims[stage_idx] * 4
                        ),
                        act=nn.GELU(),
                        pwconv2=nn.Linear(
                            stage_dims[stage_idx] * 4, stage_dims[stage_idx]
                        ),
                        drop_path=nn.Identity(),
                    )
                )
            self.stages.append(nn.Sequential(*blocks))

        # Final normalization (adjust dimension to match last stage channels)
        self.norm = nn.LayerNorm(512)  # since the final stage now has 512 channels

    def forward(self, x):
        # x: raw waveform or preprocessed spectrogram input
        # Process through extractor, etc.
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = self.spec_augmenter(x)
        x = self.speed_perturb(x)
        x = self.bn0(x)

        # Pass x through the downsample layers sequentially.
        for layer in self.downsample_layers:
            x = layer(x)

        # Pass through each stage
        for stage in self.stages:
            x = stage(x)

        # Flatten spatial dimensions (assume x is [B, C, H, W])
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # shape [B, tokens, C]
        x = self.norm(x)
        return x


# Example: create a custom projection layer mapping new encoder output dim (512) to decoder hidden size (256)
class CustomProjection(nn.Module):
    def __init__(self, in_dim=512, out_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.proj(x))


# Main training function incorporating the student model with the custom encoder
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/baseline/"
    print("Loading teacher")
    teacher_config = CoNeTTEConfig.from_pretrained(model_path)
    teacher_model = CoNeTTEModel.from_pretrained(model_path, config=teacher_config)
    teacher_model.to(device).eval()

    print("Defining student")
    # Load teacher config as baseline and then modify for a smaller encoder.
    # Option A: If CoNeTTEConfig allows setting encoder parameters, do it via the config.
    # Option B: Instantiate the teacher and then swap out the encoder.
    student_config = CoNeTTEConfig.from_pretrained(model_path)

    # Here, we will replace the teacher’s encoder with our custom SmallConvNeXtEncoder.
    # Create a student model using the teacher config first.
    student_model = CoNeTTEModel(student_config)
    # Replace encoder with our smaller version.
    student_model.preprocessor.encoder = SmallConvNeXtEncoder(
        teacher_model.preprocessor.encoder
    )

    # Replace projection layer to match new encoder output dimension (512 instead of 768)
    student_model.model.projection = nn.Sequential(
        nn.Dropout(p=0.5),
        CustomProjection(
            in_dim=512, out_dim=student_config.d_model
        ),  # student_config.d_model usually 256
        nn.ReLU(),
        nn.Dropout(p=0.5),
    )

    # Transfer the tokenizer from teacher (or keep the same, as appropriate)
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
            # Teacher output computed in eval mode (no grad)
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
