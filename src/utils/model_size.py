import torch
import torch.nn as nn
import os
from typing import Optional
from conette import CoNeTTEModel
from msclap import CLAP
from utils import config

# Sentinel to distinguish "not provided" from None (which means "skip this component")
_UNSET = object()


def get_model_size(model: torch.nn.Module, location: str = "DISK") -> float:
    """
    Measure model size in MB.
    CAUTION: DO NOT USE location="RAM" FOR QUANTIZED MODELS
    model.parameters() only include learnable parameters. Quantized parameters are not learnable, hence not returned, leading to an underestimation of the model size in RAM.
    Measuring model size on disk is more accurate, but it may not reflect the actual memory usage of the model during inference.
    Also, quantized parameters are not counted by model.parameters() and model.buffers(), so measuring their size in RAM will not reflect their actual memory usage.

    :param model: the model to measure
    :type model: torch.nn.Module
    :param location: whether to measure the size of the model in RAM or on disk
    :type location: str
    """
    if location == "RAM":
        # size of all learnable parameters
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        # size of all buffers (non-learnable parameters like running mean or variance in batch norm layers)
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size  # in bytes
        return total_size / (1024**2)  # Convert to MB
    elif location == "DISK":
        torch.save(model.state_dict(), "temp.p")
        size_mb_full = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return size_mb_full
    else:
        raise ValueError("location must be either 'RAM' or 'DISK'")


def get_model_params(model: torch.nn.Module):
    """
    Count the number of parameters in the model.
    CAUTION: DO NOT USE ON QUANTIZED MODELS

    Quantized parameters are not counted by model.parameters(), so this will not reflect the actual number of parameters in the model.
    Quantization doesn't change the amount of parameters anyway.

    :param model: the model to measure
    :type model: torch.nn.Module
    """
    return sum(p.numel() for p in model.parameters())


def count_flops(run_fn) -> int:
    """
    Count FLOPs via torch.utils.flop_counter.FlopCounterMode (PyTorch 2.0+).

    Operates at the aten operator level — captures all matmul/conv/linear ops
    including attention matmuls (Q@K^T, attn@V) and autoregressive decode steps.

    Note: dynamic quantization changes bit-width but not operation count, so
    FLOPs are expected to be identical between FP32 and INT8 variants.

    Returns total FLOPs as int.
    """
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad():
        with FlopCounterMode(display=False) as flop_counter:
            run_fn()
    return flop_counter.get_total_flops()


def measure_flops_conette(
    model: CoNeTTEModel, loader, task: str = _UNSET, n_samples: int = _UNSET
) -> Optional[int]:
    """
    Measure mean FLOPs for CoNeTTE inference over n_samples real batches.
    Caption length varies per sample, so averaging reduces variance.
    Returns mean FLOPs as int, or None on failure.
    """
    if task is _UNSET:
        task = config.dataset
    if n_samples is _UNSET:
        n_samples = config.flops_n_samples
    try:
        model.eval()
        flops_list = []
        for i, batch in enumerate(loader):
            if i >= n_samples:
                break
            audio = batch["audio"]
            sr = batch["sr"]
            flops_list.append(count_flops(lambda: model(audio, sr, task=task)))
        mean_flops = int(sum(flops_list) / len(flops_list))
        print(
            f"[FLOPs] CoNeTTE: {mean_flops / 1e9:.2f} GFLOPs "
            f"(mean over {len(flops_list)} samples)"
        )
        return mean_flops
    except Exception as e:
        print(f"[FLOPs] CoNeTTE measurement failed: {e}")
        return None


def measure_flops_clapcap(
    model: CLAP, audio_paths: list, n_samples: int = _UNSET
) -> Optional[int]:
    """
    Measure mean FLOPs for CLAPCAP inference over n_samples real audio files.
    Caption length varies per sample (autoregressive), so averaging reduces variance.

    HTSAT (Swin Transformer) uses ChunkTensor, a custom Tensor subclass that
    conflicts with FlopCounterMode's __torch_dispatch__. To work around this,
    HTSAT is run outside FlopCounterMode and measured via lightweight Conv2d/Linear
    forward hooks instead (MACs * 2 = FLOPs). Its output embedding is then fed
    directly into the Mapper and GPT-2 beam search, which are measured with
    FlopCounterMode and capture all autoregressive decode steps.

    Returns mean total FLOPs as int, or None on failure.
    """
    if n_samples is _UNSET:
        n_samples = config.flops_n_samples
    try:
        clapcap = model.clapcap
        args = model.args
        clapcap.clap.eval()
        clapcap.clap_project.eval()
        clapcap.gpt.eval()

        # HTSAT FLOPs are input-length-independent (encoder, runs once per sample).
        # Measure once via hooks and reuse for all samples.
        htsat_macs = [0]

        def _htsat_hook(module, inputs, output):
            try:
                if isinstance(module, nn.Linear):
                    x = inputs[0]
                    htsat_macs[0] += (x.numel() // x.shape[-1]) * module.in_features * module.out_features
                elif isinstance(module, nn.Conv2d):
                    x = inputs[0]
                    oh, ow = output.shape[-2], output.shape[-1]
                    htsat_macs[0] += (
                        x.shape[0] * module.out_channels * oh * ow
                        * (module.in_channels // module.groups)
                        * module.kernel_size[0] * module.kernel_size[1]
                    )
            except Exception:
                pass

        first_audio = model.preprocess_audio([audio_paths[0]], resample=True)
        handles = [
            m.register_forward_hook(_htsat_hook)
            for m in clapcap.clap.modules()
            if isinstance(m, (nn.Linear, nn.Conv2d))
        ]
        try:
            with torch.no_grad():
                clapcap.clap(first_audio.squeeze(1))
        finally:
            for h in handles:
                h.remove()
        htsat_flops = htsat_macs[0] * 2  # MACs → FLOPs

        # Measure Mapper + GPT-2 decoder FLOPs over n_samples
        decoder_flops_list = []
        caption_lengths = []
        for audio_path in audio_paths[:n_samples]:
            audio_tensors = model.preprocess_audio([audio_path], resample=True)
            with torch.no_grad():
                prefix = clapcap.clap(audio_tensors.squeeze(1))[0]
                if args.normalize_prefix:
                    prefix = prefix / prefix.norm(2, -1).reshape(-1, 1)
                prefix_embed = clapcap.clap_project(prefix).view(
                    -1, args.prefix_length, clapcap.gpt.transformer.wte.weight.shape[1]
                )
                caption = model._generate_beam(
                    embed=prefix_embed[0].unsqueeze(0),
                    beam_size=5,
                    entry_length=67,
                    temperature=1.0,
                )[0]
                caption_lengths.append(len(caption.split()))

            def run_decoder():
                clapcap.clap_project(prefix)
                model._generate_beam(
                    embed=prefix_embed[0].unsqueeze(0),
                    beam_size=5,
                    entry_length=67,
                    temperature=1.0,
                )

            decoder_flops_list.append(count_flops(run_decoder))

        mean_decoder_flops = int(sum(decoder_flops_list) / len(decoder_flops_list))
        mean_total_flops = htsat_flops + mean_decoder_flops
        mean_cap_len = sum(caption_lengths) / len(caption_lengths)
        print(
            f"[FLOPs] CLAPCAP: {mean_total_flops / 1e9:.2f} GFLOPs "
            f"(mean over {len(decoder_flops_list)} samples, avg caption {mean_cap_len:.1f} words) "
            f"[HTSAT: {htsat_flops / 1e9:.2f}, Mapper+GPT-2: {mean_decoder_flops / 1e9:.2f}]"
        )
        return mean_total_flops
    except Exception as e:
        print(f"[FLOPs] CLAPCAP measurement failed: {e}")
        return None


def count_qlinear_weight_bias_elems(qmodel: nn.Module):
    qlinear_type = torch.ao.nn.quantized.dynamic.Linear
    w_elems = 0
    b_elems = 0
    n_layers = 0
    for _, mod in qmodel.named_modules():
        if isinstance(mod, qlinear_type):
            w, b = mod._packed_params._weight_bias()
            w_elems += w.numel()
            if b is not None:
                b_elems += b.numel()
            n_layers += 1
    return w_elems, b_elems, n_layers
