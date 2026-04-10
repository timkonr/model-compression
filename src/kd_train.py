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


def train_step(student_plm, teacher_plm, batch, alpha, temperature, ce_scale, kd_scale):
    """
    Combined CE + KD loss with normalized weighting.

    Raw CE and KD losses have very different magnitudes (CE ~4-5, KD ~0.05),
    so alpha does not reflect the true gradient contribution without normalization.
    We divide each loss by a running scale estimate so alpha becomes a true weight:
        loss = (1-alpha) * (CE / ce_scale) + alpha * (KD / kd_scale)
    Scale tensors are updated in-place by the caller (EMA of observed loss values).
    """
    audio, audio_shape = batch["audio"], batch["audio_shape"]
    caps_in, caps_out = batch["captions"][:, :-1], batch["captions"][:, 1:]

    encoder_outs = student_plm.encode_audio(audio, audio_shape)
    student_logits = student_plm.decode_audio(encoder_outs, "forcing", caps_in=caps_in)
    loss_ce = student_plm.train_criterion(student_logits, caps_out)

    with torch.no_grad():
        teacher_enc = teacher_plm.encode_audio(audio, audio_shape)
        teacher_logits = teacher_plm.decode_audio(teacher_enc, "forcing", caps_in=caps_in)

    loss_kd = kd_loss(student_logits, teacher_logits, temperature)
    loss = (1.0 - alpha) * (loss_ce / ce_scale) + alpha * (loss_kd / kd_scale)
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


def train(teacher, student, dataset_name, num_epochs, batch_size, grad_accum_steps, lr, lr_encoder, alpha, temperature, patience, save_dir):
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

    # Unfreeze full student — encoder with lower lr, decoder+projection with higher lr
    # ConvNeXt encoder sits in student.preprocessor.encoder (not student.model.encoder)
    for p in student.parameters():
        p.requires_grad_(True)

    encoder_params = list(student.preprocessor.encoder.parameters())
    decoder_params = (
        list(student.model.decoder.parameters()) +
        list(student.model.projection.parameters())
    )

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    effective_batch = batch_size * grad_accum_steps
    print(f"KD training | alpha={alpha} | T={temperature} | lr_decoder={lr} | lr_encoder={lr_encoder} | bs={batch_size} (effective={effective_batch})")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"  encoder params: {sum(p.numel() for p in encoder_params):,} @ lr={lr_encoder:.1e}")
    print(f"  decoder+proj params: {sum(p.numel() for p in decoder_params):,} @ lr={lr:.1e}")

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr_encoder},
        {"params": decoder_params, "lr": lr},
    ], weight_decay=1e-4)
    total_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 100)

    best_val_loss = math.inf
    epochs_no_improve = 0
    os.makedirs(save_dir, exist_ok=True)

    # Running EMA scale estimates for loss normalization.
    # Initialized to 1.0 — will adapt after first batch.
    # EMA decay=0.98 keeps scale estimates stable but responsive.
    ema_decay = 0.98
    ce_scale = torch.tensor(1.0)
    kd_scale = torch.tensor(1.0)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(num_epochs):
        student.train()
        epoch_loss = epoch_ce = epoch_kd = 0.0
        n_steps = 0

        for step, raw_batch in enumerate(train_loader):
            batch = prepare_batch(student, raw_batch, dataset_name)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss, loss_ce, loss_kd = train_step(
                    student.model, teacher.model, batch, alpha, temperature,
                    ce_scale=ce_scale.clamp(min=1e-6),
                    kd_scale=kd_scale.clamp(min=1e-6),
                )
            # Update EMA scales
            ce_scale = ema_decay * ce_scale + (1 - ema_decay) * loss_ce.cpu()
            kd_scale = ema_decay * kd_scale + (1 - ema_decay) * loss_kd.cpu()

            # Gradient accumulation: scale loss, accumulate, step every grad_accum_steps
            scaler.scale(loss / grad_accum_steps).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item()
            epoch_ce += loss_ce.item()
            epoch_kd += loss_kd.item()
            n_steps += 1

            if step % 100 == 0:
                print(f"  epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss={loss.item():.4f} ce={loss_ce.item():.4f} kd={loss_kd.item():.4f} "
                      f"ce_scale={ce_scale.item():.3f} kd_scale={kd_scale.item():.4f} "
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
                    loss, _, _ = train_step(student.model, teacher.model, batch, alpha, temperature,
                                             ce_scale=ce_scale.clamp(min=1e-6),
                                             kd_scale=kd_scale.clamp(min=1e-6))
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
        grad_accum_steps=config.grad_accum_steps,
        lr=config.lr,
        lr_encoder=config.lr_encoder,
        alpha=config.alpha,
        temperature=config.temperature,
        patience=config.patience,
        save_dir=config.kd_save_dir,
    )


if __name__ == "__main__":
    main()
