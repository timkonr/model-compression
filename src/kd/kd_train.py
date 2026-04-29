"""
mc-train-kd: Fine-tune a pruned CoNeTTE student via Knowledge Distillation.

L = alpha * L_CE(student, ground_truth) + (1 - alpha) * L_KD(student, teacher)

The student is pruned internally from the baseline — no pre-saved pruned model needed.
Teacher is always the unpruned baseline (frozen).

Usage:
    mc-train-kd --config experiments/kd/kd_example.yaml
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import copy

import torch
import torch.nn.functional as F
from aac_datasets import AudioCaps, Clotho
from aac_datasets.utils.collate import BasicCollate
from aac_metrics import evaluate as aac_evaluate
from conette import CoNeTTEConfig, CoNeTTEModel
from torch.utils.data import DataLoader

from finetune import prepare_batch
from prune import get_conette_hidden_dims, prune_conette
from utils import config

# ---------------------------------------------------------------------------
# KD-specific loss
# ---------------------------------------------------------------------------


def kd_loss(student_logits, teacher_logits, targets, pad_idx: int):
    """
    Logit-based KD loss (Hinton et al., 2015) with τ=1.0 (no temperature scaling).
    Following Minitron (Muralidharan et al., 2024): τ=1.0, KLD loss.

    Computes KD only on non-padding target positions.

    Logits shape:  (B, V, T)
    Targets shape: (B, T)
    """
    B, V, T = student_logits.shape

    s = student_logits.permute(0, 2, 1).reshape(-1, V)  # (B*T, V)
    t = teacher_logits.permute(0, 2, 1).reshape(-1, V)  # (B*T, V)
    y = targets.reshape(-1)  # (B*T,)

    valid = y != pad_idx
    if not valid.any():
        return s.new_tensor(0.0)

    s = s[valid]
    t = t[valid]

    return F.kl_div(
        F.log_softmax(s, dim=-1),
        F.softmax(t, dim=-1),
        reduction="batchmean",
    )


def train_step(
    student_plm, teacher_plm, batch, teacher_audio_batch=None, mode="pure_kd", alpha=0.5
):
    """
    Training step supporting three modes:

    - "encoder_ce": L = L_CE only. Used when only the encoder was pruned — the
                    decoder is identical to the teacher's, so KD provides no signal.
                    Teacher forward pass is skipped entirely.
    - "pure_kd"   (Minitron BP #5): L = L_KD only — no CE loss.
    - "hybrid"    (Hinton et al. 2015): L = α·L_CE + (1-α)·L_KD
                  alpha controls the CE/KD trade-off (0=pure KD, 1=pure CE).

    CE is always computed for monitoring except in pure_kd mode.

    teacher_audio_batch: dict with "audio" and "audio_shape" from the teacher's own
    (unpruned) ConvNeXt preprocessor. Ignored in encoder_ce mode.

    Returns: (loss, loss_ce, loss_kd, alpha)
    """
    audio, audio_shape = batch["audio"], batch["audio_shape"]
    caps_in, caps_out = batch["captions"][:, :-1], batch["captions"][:, 1:]

    encoder_outs = student_plm.encode_audio(audio, audio_shape)
    student_logits = student_plm.decode_audio(
        encoder_outs, "forcing", caps_in=caps_in
    ).float()
    loss_ce = student_plm.train_criterion(student_logits, caps_out)

    if mode == "encoder_ce":
        # Decoder is identical to the teacher — CE is sufficient for encoder recovery.
        # Skip teacher forward pass entirely.
        zero = loss_ce.new_tensor(0.0)
        return loss_ce, loss_ce.detach(), zero, zero

    with torch.no_grad():
        if teacher_audio_batch is not None:
            # Teacher uses its own (unpruned) ConvNeXt output → stronger guidance signal
            t_audio = teacher_audio_batch["audio"].to(audio.device)
            t_audio_shape = teacher_audio_batch["audio_shape"].to(audio_shape.device)
        else:
            t_audio, t_audio_shape = audio, audio_shape
        teacher_enc = teacher_plm.encode_audio(t_audio, t_audio_shape)
        teacher_logits = teacher_plm.decode_audio(
            teacher_enc, "forcing", caps_in=caps_in
        ).float()

    pad_idx = student_plm.tokenizer.pad_token_id
    loss_kd = kd_loss(student_logits, teacher_logits, caps_out, pad_idx)

    if mode == "pure_kd":
        loss = loss_kd
        alpha = 0.0
    else:  # hybrid: α·L_CE + (1-α)·L_KD
        loss = alpha * loss_ce + (1.0 - alpha) * loss_kd

    return loss, loss_ce.detach(), loss_kd.detach(), torch.tensor(alpha)


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
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(subset != "val"),
        collate_fn=BasicCollate(),
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )


@torch.no_grad()
def run_fense_val(student, val_loader, dataset_name):
    """
    Run beam-search inference on the val set and return FENSE score.

    FENSE (Fluency ENhanced Sentence-level scorEr) measures semantic similarity
    via SBERT + penalises disfluent outputs. Faster than SPIDEr (no Java/SPICE),
    ~20-30s on T4 for 1045 samples. Better proxy for caption quality than CIDEr
    because it captures meaning, not just n-gram overlap.
    Higher = better.
    """
    student.eval()
    predictions, references = [], []
    for batch in val_loader:
        outputs = student(batch["audio"], batch["sr"], task=dataset_name)
        predictions.extend(outputs["cands"])
        references.extend(batch["captions"])
    corpus_scores, _ = aac_evaluate(
        candidates=predictions,
        mult_references=references,
        metrics=["fense"],
        verbose=0,
    )
    return corpus_scores["fense"].item()


def train(
    teacher,
    student,
    dataset_name,
    num_epochs,
    batch_size,
    grad_accum_steps,
    lr,
    lr_encoder,
    patience,
    save_dir,
    mode="pure_kd",
    train_components="all",
    student_hidden_dims=None,
):
    valid_modes = {"pure_kd", "hybrid", "encoder_ce"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown kd mode: {mode}. Expected one of {sorted(valid_modes)}"
        )
    valid_components = {"encoder", "decoder", "all"}
    if train_components not in valid_components:
        raise ValueError(
            f"Unknown train_components: {train_components}. Expected one of {sorted(valid_components)}"
        )

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

    # Detect which components were actually pruned by comparing student vs teacher dims.
    # Only unfreeze and optimise the pruned components — unchanged parts don't need recovery.
    teacher_hidden_dims = get_conette_hidden_dims(teacher)
    if student_hidden_dims and teacher_hidden_dims:
        encoder_pruned = any(
            student_hidden_dims.get(k) != v
            for k, v in teacher_hidden_dims.items()
            if "decoder" not in k
        )
        decoder_pruned = any(
            student_hidden_dims.get(k) != v
            for k, v in teacher_hidden_dims.items()
            if "decoder" in k
        )
    else:
        # Fallback: train everything if we can't determine what changed
        encoder_pruned = decoder_pruned = True

    if encoder_pruned and not decoder_pruned:
        print(f"Pruned: encoder only  →  strategy: {mode} (decoder frozen)")
    elif decoder_pruned and not encoder_pruned:
        print(f"Pruned: decoder only  →  strategy: {mode} (encoder frozen)")
    else:
        print(f"Pruned: encoder + decoder  →  strategy: {mode}")

    # Freeze all, then selectively unfreeze only pruned components
    for p in student.parameters():
        p.requires_grad_(False)

    encoder_params = list(student.preprocessor.encoder.parameters())
    decoder_params = list(student.model.decoder.parameters()) + list(
        student.model.projection.parameters()
    )

    opt_groups = []
    if train_components in ("encoder", "all"):
        for p in encoder_params:
            p.requires_grad_(True)
        opt_groups.append({"params": encoder_params, "lr": lr_encoder})
    if train_components in ("decoder", "all"):
        for p in decoder_params:
            p.requires_grad_(True)
        opt_groups.append({"params": decoder_params, "lr": lr})

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    effective_batch = batch_size * grad_accum_steps
    print(
        f"Fine-tuning | mode={mode} | train_components={train_components} | lr_decoder={lr} | lr_encoder={lr_encoder} | bs={batch_size} (effective={effective_batch})"
    )
    print(f"KD alpha (configured): {getattr(config, 'kd_alpha', None)}")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    if train_components in ("encoder", "all"):
        print(
            f"  encoder params: {sum(p.numel() for p in encoder_params):,} @ lr={lr_encoder:.1e}"
        )
    if train_components in ("decoder", "all"):
        print(
            f"  decoder+proj params: {sum(p.numel() for p in decoder_params):,} @ lr={lr:.1e}"
        )

    optimizer = torch.optim.AdamW(opt_groups, weight_decay=1e-5)
    total_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr / 100
    )

    best_val_metric = -math.inf  # monitor is maximized (FENSE when val loader exists)
    epochs_no_improve = 0
    os.makedirs(save_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Baseline FENSE before any training — validates pruned model quality
    if val_loader is not None:
        init_fense = run_fense_val(student, val_loader, dataset_name)
        print(f"Pre-training FENSE (pruned, no KD): {init_fense:.4f}")
        best_val_metric = init_fense  # start from pruned baseline, not -inf
    else:
        init_fense = None

    for epoch in range(num_epochs):
        student.train()
        epoch_loss = epoch_ce = epoch_kd = 0.0
        n_steps = 0

        for step, raw_batch in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # prepare_batch runs ConvNeXt (preprocessor.encoder) — keep inside
                # autocast so the encoder forward pass benefits from FP16.
                batch = prepare_batch(student, raw_batch, dataset_name)

                t_audio_batch = None
                if mode != "encoder_ce" and encoder_pruned:
                    with torch.no_grad():
                        t_audio_batch = teacher.preprocessor(
                            raw_batch["audio"], raw_batch["sr"]
                        )

                loss, loss_ce, loss_kd, alpha_value = train_step(
                    student.model,
                    teacher.model,
                    batch,
                    teacher_audio_batch=t_audio_batch,
                    mode=mode,
                    alpha=getattr(config, "kd_alpha", 0.5),
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
                lrs = scheduler.get_last_lr()
                if len(lrs) == 1:
                    lr_log = f"lr={lrs[0]:.2e}"
                else:
                    lr_log = " ".join(
                        [f"lr_g{idx}={lr_val:.2e}" for idx, lr_val in enumerate(lrs)]
                    )
                print(
                    f"  epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} ce={loss_ce.item():.4f} kd={loss_kd.item():.4f} "
                    f"alpha={alpha_value.item():.4f} {lr_log}"
                )

        avg_loss = epoch_loss / n_steps
        print(
            f"Epoch {epoch}: avg_loss={avg_loss:.4f} "
            f"avg_ce={epoch_ce/n_steps:.4f} avg_kd={epoch_kd/n_steps:.4f}"
        )

        monitor = -avg_loss  # fallback if no val set (negated: higher = better)
        val_loss = None
        val_fense = None
        if val_loader is not None:
            student.eval()
            val_loss_sum, n_val = 0.0, 0
            with torch.no_grad():
                for raw_batch in val_loader:
                    batch = prepare_batch(student, raw_batch, dataset_name)
                    t_audio_batch = None
                    if mode != "encoder_ce" and encoder_pruned:
                        t_audio_batch = teacher.preprocessor(
                            raw_batch["audio"], raw_batch["sr"]
                        )
                    loss, _, _, _ = train_step(
                        student.model,
                        teacher.model,
                        batch,
                        teacher_audio_batch=t_audio_batch,
                        mode=mode,
                        alpha=getattr(config, "kd_alpha", 0.5),
                    )
                    val_loss_sum += loss.item()
                    n_val += 1
            val_loss = val_loss_sum / max(n_val, 1)
            val_fense = run_fense_val(student, val_loader, dataset_name)
            monitor = val_fense  # higher = better
            print(f"  val_loss={val_loss:.4f}  val_fense={val_fense:.4f}")
        is_best = monitor > best_val_metric
        if is_best:
            best_val_metric = monitor
            epochs_no_improve = 0
            best_path = os.path.join(save_dir, "best")
            os.makedirs(best_path, exist_ok=True)
            # Save only the state dict — the config does not reflect pruned dimensions,
            # so from_pretrained would cause shape mismatches on reload.
            # Loading requires: rebuild_conette_from_dims() first, then load_state_dict().
            torch.save(
                student.state_dict(), os.path.join(best_path, "pytorch_model.bin")
            )
            with open(os.path.join(best_path, "meta.json"), "w") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "val_fense": monitor,
                        "mode": mode,
                        "mode_configured": mode,
                        "train_components": train_components,
                        "temperature": 1.0,
                        "alpha": getattr(config, "kd_alpha", None),
                        "lr": lr,
                        "lr_encoder": lr_encoder,
                        "dataset": dataset_name,
                        "encoder_pruned": encoder_pruned,
                        "decoder_pruned": decoder_pruned,
                        "trainable_params": {
                            "trainable": trainable,
                            "total": total,
                            "encoder": (
                                sum(p.numel() for p in encoder_params)
                                if encoder_pruned
                                else 0
                            ),
                            "decoder_projection": (
                                sum(p.numel() for p in decoder_params)
                                if decoder_pruned
                                else 0
                            ),
                        },
                        "pruning": {
                            "global_pruning_ratio": getattr(
                                config, "global_pruning_ratio", None
                            ),
                            "decoder_pruning_ratio": config.decoder_pruning_ratio,
                            "convnext_3072_threshold": config.convnext_3072_threshold,
                            "convnext_1536_threshold": config.convnext_1536_threshold,
                            "score_mode": config.pruning_score_mode,
                            "num_calibration_batches": config.num_calibration_batches,
                        },
                        "hidden_dims": student_hidden_dims,
                    },
                    f,
                    indent=2,
                )
            print(f"  => New best saved (val_fense={monitor:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")

        with open(os.path.join(save_dir, "training_log.jsonl"), "a") as log_f:
            json.dump(
                {
                    "epoch": epoch,
                    "train_loss": round(avg_loss, 6),
                    "train_ce": round(epoch_ce / n_steps, 6),
                    "train_kd": round(epoch_kd / n_steps, 6),
                    "val_loss": round(val_loss, 6) if val_loss is not None else None,
                    "val_fense": round(val_fense, 6) if val_fense is not None else None,
                    "is_best": is_best,
                    "lr": scheduler.get_last_lr(),
                },
                log_f,
            )
            log_f.write("\n")

        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

    print(f"Done. Best model at {save_dir}/best/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train pruned CoNeTTE student via KD.")
    parser.add_argument(
        "--config", required=True, help="Path to YAML experiment config."
    )
    args = parser.parse_args()

    config.load_from_yaml(args.config)
    config.set_seed(config.seed)

    baseline_path = config.model_folder + "baseline/"
    print(f"Loading baseline from {baseline_path}")
    baseline = CoNeTTEModel.from_pretrained(
        baseline_path, config=CoNeTTEConfig.from_pretrained(baseline_path)
    )
    teacher = baseline
    student = copy.deepcopy(baseline)
    print("Applying pruning to student...")
    # Build calibration loader for wanda activation-aware scoring.
    # Must use training data — never test data (data leakage).
    calib_loader = None
    if config.pruning_score_mode == "wanda":
        train_subset = "train" if config.dataset == "audiocaps" else "dev"
        calib_loader = build_dataloader(config.dataset, train_subset, batch_size=1)
    student, _ = prune_conette(student, verbose=True, loader=calib_loader)
    student_hidden_dims = get_conette_hidden_dims(student)
    print(f"Student architecture hidden dims: {student_hidden_dims}")

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
        train_components=getattr(config, "kd_train_components", "all"),
        student_hidden_dims=student_hidden_dims,
    )


if __name__ == "__main__":
    main()
