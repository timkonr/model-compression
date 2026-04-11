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

def kd_loss(student_logits, teacher_logits):
    """
    Logit-based KD loss (Hinton et al., 2015) with τ=1.0 (no temperature scaling).
    Following Minitron (Muralidharan et al., 2024): τ=1.0, KLD loss.
    Logits shape: (B, V, T) — permuted to (B*T, V) for per-token KL divergence.
    """
    B, V, T = student_logits.shape
    s = student_logits.permute(0, 2, 1).reshape(-1, V)
    t = teacher_logits.permute(0, 2, 1).reshape(-1, V)
    return F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="batchmean")


def train_step(student_plm, teacher_plm, batch, mode="pure_kd"):
    """
    KD training step supporting two modes:

    - "pure_kd"  (Minitron BP #5): L = L_KD only — no CE loss.
                 Empirically shown to outperform hybrid for pruning recovery.
    - "hybrid"   (Hinton et al. 2015): L = L_CE + alpha_dyn * L_KD, where
                 alpha_dyn = L_CE / L_KD (per-batch, keeps both terms equal magnitude).

    CE is always computed for monitoring, even in pure_kd mode.
    Returns: (loss, loss_ce, loss_kd, alpha_dyn)  — alpha_dyn=0 for pure_kd.
    """
    audio, audio_shape = batch["audio"], batch["audio_shape"]
    caps_in, caps_out = batch["captions"][:, :-1], batch["captions"][:, 1:]

    encoder_outs = student_plm.encode_audio(audio, audio_shape)
    student_logits = student_plm.decode_audio(encoder_outs, "forcing", caps_in=caps_in)
    loss_ce = student_plm.train_criterion(student_logits, caps_out)

    with torch.no_grad():
        teacher_enc = teacher_plm.encode_audio(audio, audio_shape)
        teacher_logits = teacher_plm.decode_audio(teacher_enc, "forcing", caps_in=caps_in)

    loss_kd = kd_loss(student_logits, teacher_logits)

    if mode == "pure_kd":
        loss = loss_kd
        alpha_dyn = torch.tensor(0.0)
    else:  # hybrid
        alpha_dyn = loss_ce.detach() / (loss_kd.detach() + 1e-8)
        loss = loss_ce + alpha_dyn * loss_kd

    return loss, loss_ce.detach(), loss_kd.detach(), alpha_dyn.detach()


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


def train(teacher, student, dataset_name, num_epochs, batch_size, grad_accum_steps, lr, lr_encoder, patience, save_dir, mode="pure_kd"):
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
    print(f"KD training | mode={mode} | τ=1.0 | lr_decoder={lr} | lr_encoder={lr_encoder} | bs={batch_size} (effective={effective_batch})")
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

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(num_epochs):
        student.train()
        epoch_loss = epoch_ce = epoch_kd = 0.0
        n_steps = 0

        for step, raw_batch in enumerate(train_loader):
            batch = prepare_batch(student, raw_batch, dataset_name)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss, loss_ce, loss_kd, alpha_dyn = train_step(
                    student.model, teacher.model, batch, mode=mode,
                )

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
                      f"alpha_dyn={alpha_dyn.item():.1f} lr={scheduler.get_last_lr()[0]:.2e}")

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
                    loss, _, _, _ = train_step(student.model, teacher.model, batch, mode=mode)
                    val_loss += loss.item()
                    n_val += 1
            monitor = val_loss / max(n_val, 1)
            print(f"  val_loss={monitor:.4f}")

        if monitor < best_val_loss:
            best_val_loss = monitor
            epochs_no_improve = 0
            best_path = os.path.join(save_dir, "best")
            os.makedirs(best_path, exist_ok=True)
            # Save only the state dict — the config does not reflect pruned dimensions,
            # so from_pretrained would cause shape mismatches on reload.
            # Loading requires: prune_conette() first, then load_state_dict().
            torch.save(student.state_dict(), os.path.join(best_path, "pytorch_model.bin"))
            with open(os.path.join(best_path, "meta.json"), "w") as f:
                json.dump({
                    "epoch": epoch, "val_loss": monitor,
                    "mode": mode, "temperature": 1.0,
                    "lr": lr, "dataset": dataset_name,
                    "pruning": {
                        "global_pruning_ratio": getattr(config, "global_pruning_ratio", None),
                        "convnext_3072_threshold": config.convnext_3072_threshold,
                        "convnext_1536_threshold": config.convnext_1536_threshold,
                        "decoder_threshold": config.decoder_threshold,
                        "score_mode": config.pruning_score_mode,
                        "num_calibration_batches": config.num_calibration_batches,
                    },
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
        patience=config.patience,
        save_dir=config.kd_save_dir,
        mode=config.kd_mode,
    )


if __name__ == "__main__":
    main()
