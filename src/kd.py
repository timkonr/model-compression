import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from conette import CoNeTTEModel, CoNeTTEConfig
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
from aac_metrics import evaluate

import config
from student_model import (
    extract_proj,
    EfficientNetB2AudioEncoder,
    Projection,
    pad_audio,
)

torch.backends.cudnn.benchmark = True


def validate_student(student, teacher, loader, device):
    student.eval()
    teacher.eval()
    total_loss, n = 0.0, 0
    val_bar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            audios = batch["audio"]
            t_proj = extract_proj(teacher, audios, device)
            s_proj = extract_proj(student, audios, device)
            if t_proj.size(2) != s_proj.size(2):
                t_proj = F.adaptive_avg_pool1d(t_proj, s_proj.size(2))
            total_loss += F.mse_loss(s_proj, t_proj).item()
            n += 1
    return total_loss / n


def contrastive_loss(s_proj, t_proj, alpha=0.5):
    # 1) MSE loss (as before)
    mse_loss = F.mse_loss(s_proj, t_proj)

    # 2) Utterance‐level pooling & normalization
    #   from [B, d_model, T] → [B, d_model]
    t_utt = t_proj.mean(dim=2)  # teacher utterance embeds
    s_utt = s_proj.mean(dim=2)  # student utterance embeds
    t_norm = F.normalize(t_utt, p=2, dim=1)  # [B, d_model]
    s_norm = F.normalize(s_utt, p=2, dim=1)

    # 3) InfoNCE logits & labels
    tau = 0.07
    logits = torch.matmul(s_norm, t_norm.t()) / tau  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)

    contrastive_loss = F.cross_entropy(logits, labels)

    # 4) Combined loss
    loss = alpha * mse_loss + (1 - alpha) * contrastive_loss

    return loss


def ce_loss(student_model, teacher_model, batch, device):
    """
    Args
    ----
    student_model : distilled CoNeTTEModel  (on GPU)
    teacher_model : full teacher (for tokenizer only)
    batch         : dict from BasicCollate
        ├─ "audio"     : list[Tensor]   waveforms (variable length)
        └─ "captions"  : list[str]      ground‑truth sentences
    device        : torch.device
    Returns
    -------
    loss_ce : scalar tensor
    """

    # ---------- 1) tokenise captions ---------------------------------
    tok = teacher_model.model.tokenizers["0"]
    pad_id = int(tok.pad_token_id)
    bos_id = int(tok.bos_token_id)
    eos_id = int(tok.eos_token_id)

    def _flat_ids(sent):
        """AACTokenizer may yield tensor / list[list]. Flatten to list[int]."""
        ids = tok(sent)
        if torch.is_tensor(ids):
            return ids.flatten().tolist()
        flat = []
        for x in ids if isinstance(ids, (list, tuple)) else [ids]:
            flat.extend(x.flatten().tolist() if torch.is_tensor(x) else [int(x)])
        return flat

    seqs = [[bos_id] + _flat_ids(c) + [eos_id] for c in batch["captions"]]
    max_len = max(map(len, seqs))
    teacher_ids = torch.tensor(
        [s + [pad_id] * (max_len - len(s)) for s in seqs],
        device=device,
        dtype=torch.long,
    )  # [B, L]

    # ---------- 2) encoder memory ------------------------------------
    wave, lengths = pad_audio(batch["audio"], device)
    x_shapes = torch.stack(
        (
            torch.zeros_like(torch.tensor(lengths, device=device)),
            torch.tensor(lengths, device=device),
        ),
        dim=1,
    )  # [B, 2]

    enc_out = student_model.preprocessor.encoder(wave, x_shapes)  # B,T,C
    fe = enc_out["frame_embs"].transpose(1, 2).contiguous()  # B,C,T
    mem = student_model.model.projection(fe)  # B,d,T
    mem = mem.transpose(1, 2).contiguous()  # B,T,d

    # ---------- 3) decoder teacher‑forcing ---------------------------
    dec_in = teacher_ids[:, :-1]  # input  (<bos> …)
    dec_tgt = teacher_ids[:, 1:]  # target (… <eos>)

    dec_x = student_model.model.decoder.emb_layer(dec_in)
    dec_x = student_model.model.decoder.pos_encoding(dec_x)
    dec_x = dec_x.transpose(0, 1)  # L-1,B,d
    memory = mem.transpose(0, 1)  # T,B,d

    for layer in student_model.model.decoder.layers:
        dec_x = layer(dec_x, memory)

    logits = student_model.model.decoder.classifier(dec_x.transpose(0, 1))
    B, Lm1, V = logits.size()

    loss_ce = F.cross_entropy(
        logits.reshape(-1, V),
        dec_tgt.reshape(-1),
        ignore_index=pad_id,
    )
    return loss_ce


def debug_ce(student_model, teacher_model, batch, device):
    # ---------- helper -----------------
    tok = teacher_model.model.tokenizers["0"]
    pad_id = int(tok.pad_token_id)
    bos_id, eos_id = int(tok.bos_token_id), int(tok.eos_token_id)

    def tokenize(txts):
        seqs = [[bos_id] + tok(t) + [eos_id] for t in txts]
        L = max(map(len, seqs))
        return torch.tensor([s + [pad_id] * (L - len(s)) for s in seqs], device=device)

    # ---------- prep data --------------
    audios = batch["audio"]
    gt_caps = batch["captions"]

    teacher_ids = tokenize(gt_caps)  # [B,L]

    # ---------- teacher CE -------------
    with torch.no_grad():
        loss_t = ce_loss(
            teacher_model, teacher_model, {"audio": audios, "captions": gt_caps}, device
        )
    print(" Teacher CE:", loss_t.item())

    # ---------- student CE -------------
    loss_s = ce_loss(
        student_model, teacher_model, {"audio": audios, "captions": gt_caps}, device
    )
    print(" Student CE:", loss_s.item())

    # ---------- sanity shift -----------
    print("dec_in  :", teacher_ids[0][:10].tolist())
    print("dec_tgt :", teacher_ids[0][1:11].tolist())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Teacher
    base_path = config.model_folder + "baseline/"
    teacher_cfg = CoNeTTEConfig.from_pretrained(base_path)
    teacher_model = CoNeTTEModel.from_pretrained(base_path, config=teacher_cfg)
    teacher_model.to(device).eval()

    # Student
    student_cfg = CoNeTTEConfig.from_pretrained(base_path)
    student_model = CoNeTTEModel(student_cfg)
    student_model.preprocessor.encoder = EfficientNetB2AudioEncoder(
        teacher_model.preprocessor.encoder
    )
    enc_ch = student_model.preprocessor.encoder.out_channels
    dec_ch = student_cfg.d_model
    student_model.model.projection = Projection(enc_ch, dec_ch, p=0.5)
    student_model.model.tokenizers["0"] = teacher_model.model.tokenizers["0"]
    student_model.to(device)

    # Data loaders
    train_loader = DataLoader(
        Clotho(config.data_folder, subset="dev"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=BasicCollate(),
    )
    val_loader = DataLoader(
        Clotho(config.data_folder, subset="val"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=BasicCollate(),
    )

    # Optimizer + LR schedule
    optimizer = optim.AdamW(student_model.parameters(), lr=1e-4)
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # debug information
    v_teacher = teacher_model.model.decoder.classifier.out_features
    v_student = student_model.model.decoder.classifier.out_features
    print("Vocab sizes teacher, student", v_teacher, v_student)
    with torch.no_grad():
        batch = next(iter(train_loader))
        print(
            "Initial CE loss:",
            ce_loss(student_model, teacher_model, batch, device).item(),
        )
        debug_ce(student_model, teacher_model, batch, device)

    print("Starting distillation…")

    best_val = float("inf")
    best_epoch = 0
    no_improve = 0
    lambda_feat = 0.3  # contrastive loss weight
    lambda_seq = 0.02  # sequence level KD weight
    lambda_ce = 1.0  # cross entropy loss weight

    for epoch in range(1, config.num_epochs + 1):
        student_model.train()
        total_loss = 0.0
        total_feat = 0.0
        total_seq = 0.0
        total_ce = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"[Epoch {epoch}/{config.num_epochs}] Training",
            leave=False,
        )

        for batch in train_bar:
            audios = batch["audio"]
            with torch.no_grad():
                T_out = teacher_model(audios)
                t_lprobs = T_out["lprobs"].to(device)  # [B]
                t_proj = extract_proj(teacher_model, audios, device)

            S_out = student_model(audios)
            s_lprobs = S_out["lprobs"]  # [B]

            # MSE on log‐probs is a simple sequence‐level KD
            loss_seq = F.mse_loss(s_lprobs, t_lprobs)

            loss_ce = ce_loss(student_model, teacher_model, batch, device)

            # plus your feature‐KD
            s_proj = extract_proj(student_model, audios, device)
            if t_proj.size(2) != s_proj.size(2):
                t_proj = F.adaptive_avg_pool1d(t_proj, s_proj.size(2))
            loss_feat = contrastive_loss(s_proj, t_proj)

            loss = lambda_feat * loss_feat + lambda_seq * loss_seq + lambda_ce * loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_feat += loss_feat.item()
            total_seq += loss_seq.item()
            total_ce += loss_ce.item()
        avg_train = total_loss / len(train_loader)
        avg_feat = total_feat / len(train_loader)
        avg_seq = total_seq / len(train_loader)
        avg_ce = total_ce / len(train_loader)
        val_loss = validate_student(student_model, teacher_model, val_loader, device)
        print(
            f"[Epoch {epoch}] train_loss=(feat={avg_feat:.4f}, seq={avg_seq:.4f}, ce={avg_ce:.4f}, total={avg_train:.4f})  val_loss={val_loss:.4f}"
        )

        # Evaluate student model on bleu, fense and spider-fl every 5 epochs
        if epoch % 5 == 0:
            student_model.eval()
            candidates = []
            mult_references = []
            for batch in val_loader:
                audios = batch["audio"]
                with torch.no_grad():
                    outputs = student_model(audios, task="clotho")
                    candidates.extend(outputs["cands"])
                    mult_references.extend(batch["captions"])

            evaluate(
                candidates=candidates,
                mult_references=mult_references,
                metrics=["bleu_1", "fense", "spider_fl"],
            )

        # early stopping logic
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                student_model.state_dict(),
                config.model_folder + config.kd_model,
            )
            print("  → Saved new best")
        else:
            no_improve += 1
            print(f"  → No improvement ({no_improve}/{config.patience})")

        if no_improve >= config.patience:
            print(
                f"Stopping early at epoch {epoch} (no improvement for {config.patience} epochs)."
            )
            break

    print(f"Training finished. Best val_loss={best_val:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
