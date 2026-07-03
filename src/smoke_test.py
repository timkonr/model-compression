#!/usr/bin/env python3
"""Smoke test for a rented GPU pod: verifies GPU, checkpoint save/RESUME, and
optional W&B logging. This is the reliability test before the real runs.

Test it like this:
  1) python smoke_test.py --steps 10 --ckpt-dir /workspace/ckpt   # let it run a few steps
  2) Ctrl-C  (or 'Stop' the pod)  ->  simulates an interruption
  3) python smoke_test.py --steps 10 --ckpt-dir /workspace/ckpt   # MUST resume, not restart

Add --wandb to also verify experiment tracking (needs `wandb login` first).
Put --ckpt-dir on your persistent Network Volume (e.g. /workspace/ckpt) so a
dead node loses nothing.
"""
import argparse, glob, os, time
import torch
import torch.nn as nn


def latest_ckpt(d):
    files = glob.glob(os.path.join(d, "ckpt_*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--ckpt-dir", default="./ckpt")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=1.0, help="sec per step (to give you time to Ctrl-C)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] torch={torch.__version__}  device={device}"
          + (f"  gpu={torch.cuda.get_device_name(0)}" if device == "cuda" else "  (NO GPU!)"))

    torch.manual_seed(0)
    model = nn.Linear(16, 16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_step, run_id = 0, None

    ck = latest_ckpt(args.ckpt_dir)
    if ck:
        state = torch.load(ck, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"] + 1
        run_id = state.get("wandb_run_id")
        print(f"[RESUME] loaded {os.path.basename(ck)} -> continuing at step {start_step}")
    else:
        print("[fresh] no checkpoint found -> starting at step 0")

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project="smoke-test", id=run_id, resume="allow")
        run_id = wb.id

    for step in range(start_step, args.steps):
        x = torch.randn(32, 16, device=device)
        loss = model(x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"[step {step}] loss={float(loss):.4f}")
        if wb:
            wb.log({"loss": float(loss)}, step=step)
        time.sleep(args.sleep)
        if step % args.save_every == 0:
            path = os.path.join(args.ckpt_dir, f"ckpt_{step:05d}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "wandb_run_id": run_id}, path)

    print("[done] reached target steps — resume works, pod is good to go")
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
