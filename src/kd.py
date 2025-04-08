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


def distillation_loss(student_output, teacher_output, tokenizer, transformer, device):
    # Tokenize the input sentences (if not already tokenized)
    student_tokens = tokenizer(
        student_output, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    teacher_tokens = tokenizer(
        teacher_output, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # Get embeddings from the transformer model (typically last_hidden_state)
    student_out = transformer(**student_tokens)
    teacher_out = transformer(**teacher_tokens)

    # Get the embeddings (mean pooling over the token embeddings)
    student_embed = mean_pooling(student_out, student_tokens["attention_mask"])
    with torch.no_grad():  # Freeze teacher embeddings
        teacher_embed = mean_pooling(teacher_out, teacher_tokens["attention_mask"])

    # Check that the embeddings have the same shape
    print(f"Student Embed Shape: {student_embed.shape}, embedding: {student_embed}")
    print(f"Teacher Embed Shape: {teacher_embed.shape}, embedding: {teacher_embed}")

    # Compute MSE loss between student and teacher embeddings
    loss = nn.functional.mse_loss(student_embed, teacher_embed)

    return loss


def validate_student(model, val_loader, tokenizer, transformer, device):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["audio"]
            student_output = model(inputs)["cands"]
            teacher_output = model(inputs)[
                "cands"
            ]  # Assuming we use self-teacher for simplicity

            loss = distillation_loss(
                student_output, teacher_output, tokenizer, transformer, device
            )
            total_loss += loss.item()
            count += 1

    return total_loss / count


def train_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    tokenizer,
    transformer,
    device,
    num_epochs: int = 10,
    save_path: str = "best_student_model.pth",
):
    teacher_model.to(device).eval()
    student_model.to(device)

    best_val_loss = float("inf")

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
            student_model, val_loader, tokenizer, transformer, device
        )
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), save_path)
            print("Saved best student model.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/"
    # Load teacher (pretrained)
    print("Load teacher")
    teacher_config = CoNeTTEConfig.from_pretrained(model_path)
    teacher_model = CoNeTTEModel.from_pretrained(model_path, config=teacher_config)

    print("Define student")
    # Define a smaller student model
    student_config = CoNeTTEConfig.from_pretrained(model_path)
    student_config.hidden_size = 256
    student_config.num_attention_heads = 4
    student_config.intermediate_size = 512
    student_config.num_hidden_layers = 3
    student_model = CoNeTTEModel(student_config)
    student_model.model.tokenizers["0"] = teacher_model.model.tokenizers["0"]

    # Datasets
    print("Load data")
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

    # Train with knowledge distillation
    print("Start distillation")
    train_distillation(
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        optimizer,
        tokenizer,
        transformer,
        device,
        num_epochs=20,
        save_path="best_student_model.pth",
    )


if __name__ == "__main__":
    main()
