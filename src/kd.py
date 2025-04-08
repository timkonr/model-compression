import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from conette import CoNeTTEModel, CoNeTTEConfig
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# Mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [B, T, H]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / input_mask_expanded.sum(1)


def distillation_loss(
    student_output, teacher_output, device, alpha=0.5, temperature=2.0
):
    # Load HuggingFace backbone directly
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = AutoModel.from_pretrained(model_name).to(device)

    # Tokenize input
    student_tokens = tokenizer(
        student_output["cands"], padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    teacher_tokens = tokenizer(
        teacher_output["cands"], padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # Forward pass (grad-enabled for student)
    student_out = transformer(**student_tokens)
    teacher_out = transformer(**teacher_tokens)

    student_embed = mean_pooling(
        student_out, student_tokens["attention_mask"]
    )  # grads enabled
    with torch.no_grad():  # freeze teacher
        teacher_embed = mean_pooling(teacher_out, teacher_tokens["attention_mask"])

    # Loss
    loss = nn.functional.mse_loss(student_embed, teacher_embed)
    return loss


def train_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    alpha: float = 0.5,
    temperature: float = 2.0,
    num_epochs: int = 10,
    save_path: str = "best_student_model.pth",
):
    """
    Train the student model using full-model knowledge distillation.
    """
    teacher_model.to(device)
    student_model.to(device)

    teacher_model.eval()
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        student_model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs, labels = batch["audio"], batch["captions"]

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
                teacher_preds = teacher_outputs["preds"]

            student_outputs = student_model(inputs)
            student_preds = student_outputs["preds"]

            loss = distillation_loss(
                student_outputs, teacher_outputs, device, alpha, temperature
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Training Loss: {avg_train_loss:.4f}")

        # Validation
        val_loss = evaluate_student(student_model, val_loader, device)
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student_model.state_dict(), save_path)
            print("Saved best student model.")


def evaluate_student(model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["audio"].to(device), batch["labels"].to(device)
            outputs = model(inputs)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/"
    # Load teacher (pretrained)
    print("Load teacher")
    teacher_config = CoNeTTEConfig.from_pretrained(model_path)
    teacher_model = CoNeTTEModel.from_pretrained(model_path, config=teacher_config)

    print("Define student")
    # Define a smaller student model
    student_config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    student_config.hidden_size = 256
    student_config.num_attention_heads = 4
    student_config.intermediate_size = 512
    student_config.num_hidden_layers = 3  # Shallower
    student_model = CoNeTTEModel(student_config)
    print(teacher_model.model.tokenizers.keys())

    student_model.model.tokenizers["0"] = teacher_model.model.tokenizers["0"]

    # Datasets
    print("Load data")
    train_dataset = Clotho("data", subset="dev")
    val_dataset = Clotho("data", subset="val")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=BasicCollate(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=BasicCollate(),
    )

    # Optimizer
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-4)

    # Train with knowledge distillation
    print("Start distillation")
    train_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        alpha=0.5,
        temperature=2.0,
        num_epochs=20,
        save_path="best_student_model.pth",
    )


if __name__ == "__main__":
    main()
