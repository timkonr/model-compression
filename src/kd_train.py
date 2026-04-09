"""
mc-train-kd: Fine-tune a pruned CoNeTTE student via Knowledge Distillation.

L = (1 - alpha) * L_CE(student, ground_truth) + alpha * L_KD(student, teacher)

The student is pruned internally from the baseline — no pre-saved pruned model needed.
Teacher is always the unpruned baseline (frozen).

Usage:
    mc-train-kd --config experiments/kd_example.yaml
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F
from aac_datasets import AudioCaps, Clotho
from aac_datasets.utils.collate import BasicCollate
from conette import CoNeTTEConfig, CoNeTTEModel
from torch.utils.data import DataLoader

from finetune import prepare_batch
from prune import prune_conette
from utils import config


# ---------------------------------------------------------------------------
# KD-specific loss
# ---------------------------------------------------------------------------

def kd_loss(student_logits, teacher_logits, temperature):
    """
    Logit-based KD loss (Hinton et al., 2015).
    Logits shape: (B, V, T) — permuted to (B*T, V) for per-token KL divergence.
    Scaled by T² to keep gradient magnitude comparable across temperatures.
    """
    B, V, T = student_logits.shape
    s = student_logits.permute(0, 2, 1).reshape(-1, V) / temperature
    t = teacher_logits.permute(0, 2, 1).reshape(-1, V) / temperature
    loss = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean")
    return loss * (temperature ** 2)


def train_step(student_plm, teacher_plm, batch, alpha, temperature):
    audio, audio_shape = batch["audio"], batch["audio_shape"]
    caps_in, caps_out = batch["captions"][:, :-1], batch["captions"][:, 1:]

    encoder_outs = student_plm.encode_audio(audio, audio_shape)
    student_logits = student_plm.decode_audio(encoder_outs, "forcing", caps_in=caps_in)
    loss_ce = student_plm.train_criterion(student_logits, caps_out)

    with torch.no_grad():
        teacher_enc = teacher_plm.encode_audio(audio, audio_shape)
        teacher_logits = teacher_plm.decode_audio(teacher_enc, "forcing", caps_in=caps_in)

    loss_kd = kd_loss(student_logits, teacher_logits, temperature)
    loss = (1.0 - alpha) * loss_ce + alpha * loss_kd
    return loss, loss_ce.detach(), loss_kd.detach()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def build_dataloader(dataset_name, subset, batch_size):
    if dataset_name == "audiocaps":
        ds = AudioCaps(config.data_folder, subset=subset, audio_format="wav", sr=22050)
    elif dataset_name == "clotho":
        ds = Clotho(config.data_folder, subset=subset)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DataLoader(ds, batch_size=batch_size, shuffle=(subset != "val"), collate_fn=BasicCollate())


def train(teacher, student, dataset_name, num_epochs, batch_size, lr, alpha, temperature, patience, save_dir):
    train_subset = "train" if dataset_name == "audiocaps" else "dev"

    train_loader = build_dataloader(dataset_name, train_subset, batch_size)
    try:
        val_loader = build_dataloader(dataset_name, "val", batch_size)
    except Exception:
        val_loader = None

    # Freeze teacher entirely
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # Freeze student encoder, unfreeze decoder + projection
    for p in student.parameters():
        p.requires_grad_(False)
    for p in student.model.decoder.parameters():
        p.requires_grad_(True)
    for p in student.model.projection.parameters():
        p.requires_grad_(True)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    print(f"KD training | alpha={alpha} | T={temperature} | lr={lr} | bs={batch_size}")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    total_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 100)

    best_val_loss = math.inf
    epochs_no_improve = 0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        student.train()
        epoch_loss = epoch_ce = epoch_kd = 0.0
        n_steps = 0

        for step, raw_batch in enumerate(train_loader):
            batch = prepare_batch(student, raw_batch, dataset_name)
            optimizer.zero_grad()
            loss, loss_ce, loss_kd = train_step(
                student.model, teacher.model, batch, alpha, temperature
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_ce += loss_ce.item()
            epoch_kd += loss_kd.item()
            n_steps += 1

            if step % 100 == 0:
                print(f"  epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss={loss.item():.4f} ce={loss_ce.item():.4f} kd={loss_kd.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / n_steps
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f} "
              f"avg_ce={epoch_ce/n_steps:.4f} avg_kd={epoch_kd/n_steps:.4f}")

        # Validation
        monitor = avg_loss
        if val_loader is not None:
            student.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for raw_batch in val_loader:
                    batch = prepare_batch(student, raw_batch, dataset_name)
                    loss, _, _ = train_step(student.model, teacher.model, batch, alpha, temperature)
                    val_loss += loss.item()
                    n_val += 1
            monitor = val_loss / max(n_val, 1)
            print(f"  val_loss={monitor:.4f}")

        if monitor < best_val_loss:
            best_val_loss = monitor
            epochs_no_improve = 0
            best_path = os.path.join(save_dir, "best")
            student.save_pretrained(best_path)
            with open(os.path.join(best_path, "meta.json"), "w") as f:
                json.dump({
                    "epoch": epoch, "val_loss": monitor,
                    "alpha": alpha, "temperature": temperature,
                    "lr": lr, "dataset": dataset_name,
                }, f, indent=2)
            print(f"  => New best saved (loss={monitor:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    print(f"Done. Best model at {save_dir}/best/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train pruned CoNeTTE student via KD.")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    args = parser.parse_args()

    config.load_from_yaml(args.config)
    config.set_seed(config.seed)

    teacher_path = config.model_folder + "baseline/"
    print(f"Loading teacher from {teacher_path}")
    teacher = CoNeTTEModel.from_pretrained(
        teacher_path, config=CoNeTTEConfig.from_pretrained(teacher_path)
    )

    print("Loading student and applying pruning...")
    student = CoNeTTEModel.from_pretrained(
        teacher_path, config=CoNeTTEConfig.from_pretrained(teacher_path)
    )
    student, _ = prune_conette(student, verbose=True)

    train(
        teacher=teacher,
        student=student,
        dataset_name=config.dataset,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        alpha=config.alpha,
        temperature=config.temperature,
        patience=config.patience,
        save_dir=config.kd_save_dir,
    )


if __name__ == "__main__":
    main()
