import torch
import torch.nn as nn
from typing import Optional
import utils.config as config
from transformers.pytorch_utils import Conv1D
from conette import CoNeTTEModel

# Sentinel to distinguish "not provided" from None (which means "skip this component")
_UNSET = object()


def compute_linear_hidden_scores(
    first: nn.Linear,
    second: nn.Linear,
    mode: str = "sum_l2",
) -> torch.Tensor:
    """
    Compute one importance score per hidden neuron.

    first:  Linear(in_features, hidden_dim)
    second: Linear(hidden_dim, out_features)

    Shapes:
      first.weight  = (hidden_dim, in_features)
      second.weight = (out_features, hidden_dim)
    """
    if mode == "random":
        return torch.rand(first.out_features)

    if mode == "first_l2":
        return first.weight.norm(p=2, dim=1)

    if mode == "sum_l2":
        # importance = incoming magnitude + outgoing magnitude
        in_score = first.weight.norm(p=2, dim=1)  # (hidden_dim,)
        out_score = second.weight.norm(p=2, dim=0)  # (hidden_dim,)
        return in_score + out_score

    raise ValueError(f"Unsupported score mode: {mode}")


@torch.no_grad()
def prune_linear_pair(
    first: nn.Linear,
    second: nn.Linear,
    keep_ratio: float,
    score_mode: str = "sum_l2",
) -> tuple[nn.Linear, nn.Linear]:
    """
    Structured pruning of an MLP pair:

      first : Linear(d_in, d_hidden)
      second: Linear(d_hidden, d_out)

    Removes hidden neurons consistently in both layers.
    """
    d_hidden = first.out_features
    d_in = first.in_features
    d_out = second.out_features

    if second.in_features != d_hidden:
        raise RuntimeError(
            f"Incompatible pair: first.out={d_hidden}, second.in={second.in_features}"
        )

    k = int(round(d_hidden * keep_ratio))
    k = max(1, min(k, d_hidden))

    keep_idx = torch.topk(scores, k=k, largest=True).indices
    keep_idx, _ = torch.sort(keep_idx)

    new_first = nn.Linear(d_in, k, bias=(first.bias is not None)).to(
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    new_second = nn.Linear(k, d_out, bias=(second.bias is not None)).to(
        device=second.weight.device,
        dtype=second.weight.dtype,
    )

    new_first.weight.copy_(first.weight[keep_idx, :])
    if first.bias is not None:
        new_first.bias.copy_(first.bias[keep_idx])

    new_second.weight.copy_(second.weight[:, keep_idx])
    if second.bias is not None:
        new_second.bias.copy_(second.bias)

    return new_first, new_second


@torch.no_grad()
def prune_conette(
    model: CoNeTTEModel,
    decoder_keep_ratio: Optional[float] = _UNSET,
    convnext_3072_keep_ratio: Optional[float] = _UNSET,
    convnext_1536_keep_ratio: Optional[float] = _UNSET,
    score_mode: str = _UNSET,
    verbose: bool = True,
    loader: torch.utils.data.DataLoader = None,
):
    """
    Prune CoNeTTE MLP blocks:
      - decoder: linear1 / linear2
      - encoder: pwconv1 / pwconv2

    Set a keep_ratio to None to skip that part.
    """
    if decoder_keep_ratio is _UNSET:
        decoder_keep_ratio = config.decoder_keep_ratio
    if convnext_3072_keep_ratio is _UNSET:
        convnext_3072_keep_ratio = config.convnext_3072_keep_ratio
    if convnext_1536_keep_ratio is _UNSET:
        convnext_1536_keep_ratio = config.convnext_1536_keep_ratio
    if score_mode is _UNSET:
        score_mode = config.pruning_score_mode

    model.eval()

    pruned_layer_names = set()

    # -------------------------
    # 1) Decoder pruning
    # -------------------------
    if decoder_keep_ratio is not None:
        dec = model.model.decoder
        for li, layer in enumerate(dec.layers):
            old_hidden = layer.linear1.out_features

            new_fc1, new_fc2 = prune_linear_pair(
                layer.linear1,
                layer.linear2,
                keep_ratio=decoder_keep_ratio,
                score_mode=score_mode,
            )

            layer.linear1 = new_fc1
            layer.linear2 = new_fc2

            pruned_layer_names.add(f"model.decoder.layers.{li}.linear1")
            pruned_layer_names.add(f"model.decoder.layers.{li}.linear2")

            if verbose:
                print(
                    f"[Decoder layer {li}] hidden dim: "
                    f"{old_hidden} -> {new_fc1.out_features}"
                )

    # -------------------------
    # 2) Encoder pruning
    # -------------------------
    if convnext_3072_keep_ratio is not None or convnext_1536_keep_ratio is not None:
        for stage_idx, stage in enumerate(model.preprocessor.encoder.stages):
            for block_idx, block in enumerate(stage):
                if not (hasattr(block, "pwconv1") and hasattr(block, "pwconv2")):
                    continue

                pw1 = block.pwconv1
                pw2 = block.pwconv2

                if not isinstance(pw1, nn.Linear) or not isinstance(pw2, nn.Linear):
                    continue

                old_hidden = pw1.out_features
                keep_ratio = None

                if old_hidden == 1536 and convnext_1536_keep_ratio is not None:
                    keep_ratio = convnext_1536_keep_ratio
                elif old_hidden == 3072 and convnext_3072_keep_ratio is not None:
                    keep_ratio = convnext_3072_keep_ratio

                if keep_ratio is None:
                    continue

                new_pw1, new_pw2 = prune_linear_pair(
                    pw1,
                    pw2,
                    keep_ratio=keep_ratio,
                    score_mode=score_mode,
                )

                block.pwconv1 = new_pw1
                block.pwconv2 = new_pw2

                base_name = f"preprocessor.encoder.stages.{stage_idx}.{block_idx}"
                pruned_layer_names.add(f"{base_name}.pwconv1")
                pruned_layer_names.add(f"{base_name}.pwconv2")

                if verbose:
                    print(
                        f"[ConvNeXt stage {stage_idx} block {block_idx}] hidden dim: "
                        f"{old_hidden} -> {new_pw1.out_features}"
                    )

    return model, pruned_layer_names


def compute_conv1d_hidden_scores(
    first: Conv1D,
    second: Conv1D,
    mode: str = "sum_l2",
) -> torch.Tensor:
    """
    first : Conv1D(hidden_dim, in_features)
    second: Conv1D(out_features, hidden_dim)

    Shapes:
        first.weight  = (in_features, hidden_dim)
        second.weight = (hidden_dim, out_features)
    """
    if mode == "random":
        return torch.rand(first.weight.shape[1])

    if mode == "first_l2":
        return first.weight.norm(p=2, dim=0)  # columns of first => hidden dim

    if mode == "sum_l2":
        in_score = first.weight.norm(p=2, dim=0)  # columns of first => hidden dim
        out_score = second.weight.norm(p=2, dim=1)  # rows of second => hidden dim
        return in_score + out_score

    raise ValueError(f"Unknown score mode: {mode}")


@torch.no_grad()
def prune_conv1d_pair(
    first: Conv1D, second: Conv1D, keep_ratio: float, score_mode: str = "sum_l2"
):
    d_in, d_hidden = first.weight.shape
    d_hidden_2, d_out = second.weight.shape

    if d_hidden_2 != d_hidden:
        raise RuntimeError(
            f"Incompatible Conv1D pair: first hidden={d_hidden}, second hidden={d_hidden_2}"
        )

    k = int(round(d_hidden * keep_ratio))
    k = max(1, min(k, d_hidden))

    scores = compute_conv1d_hidden_scores(first, second, mode=score_mode)
    keep_idx = torch.topk(scores, k=k, largest=True).indices
    keep_idx, _ = torch.sort(keep_idx)

    new_first = Conv1D(k, d_in).to(device=first.weight.device, dtype=first.weight.dtype)
    new_second = Conv1D(d_out, k).to(
        device=second.weight.device, dtype=second.weight.dtype
    )

    # first.weight: (in_features, hidden_dim) -> keep hidden columns
    new_first.weight.copy_(first.weight[:, keep_idx])
    new_first.bias.copy_(first.bias[keep_idx])

    # second.weight: (hidden_dim, out_features) -> keep hidden rows
    new_second.weight.copy_(second.weight[keep_idx, :])
    new_second.bias.copy_(second.bias)

    return new_first, new_second


@torch.no_grad()
def prune_clapcap(
    model: nn.Module,
    gpt_keep_ratio: Optional[float] = _UNSET,
    mapper_keep_ratio: Optional[float] = _UNSET,
    htsat_keep_ratio: Optional[float] = _UNSET,
    htsat_min_hidden_dim: int = _UNSET,
    score_mode: str = "sum_l2",
    verbose: bool = True,
) -> nn.Module:
    """
    Structured pruning for CLAPCAP on:
      - GPT2 MLPs        ("Conv1D" pair) -> HuggingFace's Conv1D basically works like a linear layer but the weights are transposed.
      - Mapper MLPs      (Linear pair)
      - HTSAT/Swin MLPs  (Linear pair)
    """
    if gpt_keep_ratio is _UNSET:
        gpt_keep_ratio = config.gpt_keep_ratio
    if mapper_keep_ratio is _UNSET:
        mapper_keep_ratio = config.mapper_keep_ratio
    if htsat_keep_ratio is _UNSET:
        htsat_keep_ratio = config.htsat_keep_ratio
    if htsat_min_hidden_dim is _UNSET:
        htsat_min_hidden_dim = config.htsat_min_hidden_dim

    model.eval()

    # -------------------------
    # 1) GPT2 pruning
    # -------------------------
    if gpt_keep_ratio is not None:
        gpt_blocks = model.gpt.transformer.h

        for i, block in enumerate(gpt_blocks):
            old_hidden = block.mlp.c_fc.weight.shape[1]

            new_c_fc, new_c_proj = prune_conv1d_pair(
                block.mlp.c_fc,
                block.mlp.c_proj,
                keep_ratio=gpt_keep_ratio,
                score_mode=score_mode,
            )

            block.mlp.c_fc = new_c_fc
            block.mlp.c_proj = new_c_proj

            if verbose:
                print(
                    f"[GPT2 block {i}] hidden dim: {old_hidden} -> {new_c_fc.weight.shape[1]}"
                )
    # -------------------------
    # 2) Mapper pruning
    # -------------------------
    if mapper_keep_ratio is not None:
        mapper_layers = model.clap_project.transformer.layers

        for i, layer in enumerate(mapper_layers):
            old_hidden = layer.mlp.fc1.out_features

            new_fc1, new_fc2 = prune_linear_pair(
                layer.mlp.fc1,
                layer.mlp.fc2,
                keep_ratio=mapper_keep_ratio,
                score_mode=score_mode,
            )

            layer.mlp.fc1 = new_fc1
            layer.mlp.fc2 = new_fc2

            if verbose:
                print(
                    f"[Mapper layer {i}] hidden dim: {old_hidden} -> {new_fc1.out_features}"
                )

    # -------------------------
    # 3) HTSAT / Swin MLP pruning
    # -------------------------
    if htsat_keep_ratio is not None:
        global_block_idx = 0

        for stage_idx, stage in enumerate(model.clap.base.htsat.layers):
            if not hasattr(stage, "blocks"):
                continue

            for block_idx, block in enumerate(stage.blocks):
                current_block_idx = global_block_idx
                global_block_idx += 1
                if not hasattr(block, "mlp"):
                    continue

                fc1 = block.mlp.fc1
                fc2 = block.mlp.fc2

                if fc1.out_features < htsat_min_hidden_dim:
                    continue

                old_hidden = fc1.out_features

                new_fc1, new_fc2 = prune_linear_pair(
                    fc1,
                    fc2,
                    keep_ratio=htsat_keep_ratio,
                    score_mode=score_mode,
                )

                block.mlp.fc1 = new_fc1
                block.mlp.fc2 = new_fc2

                if verbose:
                    print(
                        f"[HTSAT stage {stage_idx} block {block_idx} | global {current_block_idx}] "
                        f"hidden dim: {old_hidden} -> {new_fc1.out_features}"
                    )

    return model
