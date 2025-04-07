import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from conette import CoNeTTEModel, CoNeTTEConfig
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate

# from loss import CrossEntropyLoss


def distillation_loss(
    student_preds, teacher_preds, ground_truth, alpha=0.5, temperature=2.0
):
    # Soft Targets Loss
    teacher_probs = nn.functional.softmax(teacher_preds / temperature, dim=-1)
    student_probs = nn.functional.softmax(student_preds / temperature, dim=-1)
    # kd_loss = nn.functional.kl_div(
    #     student_probs, teacher_probs, reduction="batchmean"
    # ) * (temperature**2)

    # # Hard Targets Loss (cross-entropy loss for ground-truth)
    # ce_loss = nn.CrossEntropyLoss()(
    #     student_preds.view(-1, student_preds.size(-1)), ground_truth.view(-1)
    # )

    # # Combine losses
    # return alpha * kd_loss + (1 - alpha) * ce_loss
    # Compute KL divergence between the student and teacher probabilities
    kd_loss = nn.functional.F.kl_div(
        student_probs.log(), teacher_probs, reduction="batchmean"
    )
    return kd_loss


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
                student_preds, teacher_preds, labels, alpha, temperature
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

    # Load teacher (pretrained)
    print("Load teacher")
    teacher_config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    teacher_model = CoNeTTEModel.from_pretrained(
        "Labbeti/conette", config=teacher_config
    )

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
