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
    mode: str = "wanda",
    activation_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute one importance score per hidden neuron for a linear pair.

    first:  Linear(in_features, hidden_dim)
    second: Linear(hidden_dim, out_features)

    Shapes:
      first.weight  = (hidden_dim, in_features)
      second.weight = (out_features, hidden_dim)

    For mode="wanda", activation_scores must correspond to the input features
    of `first`, i.e. shape (in_features,).
    """
    if mode == "random":
        return torch.rand(first.out_features)

    if mode == "first_l2":
        return first.weight.norm(p=2, dim=1)

    if mode == "sum_l2":
        in_score = first.weight.norm(p=2, dim=1)
        out_score = second.weight.norm(p=2, dim=0)
        return in_score + out_score

    if mode == "wanda":
        if activation_scores is None:
            raise RuntimeError("activation_scores are required for mode='wanda'")

        if activation_scores.numel() != first.out_features:
            raise RuntimeError(
                f"Wanda scores shape {activation_scores.shape} does not match "
                f"first.out_features={first.out_features}"
            )

        return activation_scores.to(
            device=first.weight.device, dtype=first.weight.dtype
        )

    raise ValueError(f"Unsupported linear score mode: {mode}")


@torch.no_grad()
def prune_linear_pair(
    first: nn.Linear,
    second: nn.Linear,
    threshold: float,
    score_mode: str = "sum_l2",
    activation_scores: Optional[torch.Tensor] = None,
) -> tuple[nn.Linear, nn.Linear, torch.Tensor, float]:
    """
    Structured pruning of an MLP pair:

      first : Linear(d_in, d_hidden)
      second: Linear(d_hidden, d_out)

    Removes hidden neurons consistently in both layers.

    Returns:
      new_first, new_second, scores
    """
    d_hidden = first.out_features
    d_in = first.in_features
    d_out = second.out_features

    if second.in_features != d_hidden:
        raise RuntimeError(
            f"Incompatible pair: first.out={d_hidden}, second.in={second.in_features}"
        )

    threshold = max(0.0, min(1.0, float(threshold)))

    scores = compute_linear_hidden_scores(
        first,
        second,
        mode=score_mode,
        activation_scores=activation_scores,
    )

    k = int(round(d_hidden * threshold))
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


class ActivationCollector:
    """
    Collect structured Wanda scores for one linear layer pair.

    Aggregation:
      - sequence/time dimension: mean
      - batch dimension: L2
    """

    def __init__(self, first: nn.Linear):
        self.first = first
        self.score_sum = None
        self.num_updates = 0

    def update(self, tensor: torch.Tensor) -> None:
        """
        tensor shape:
          - (in_features,)
          - (batch, in_features)
          - (batch, seq, in_features)
          - or any (..., in_features)
        """
        x = (
            tensor.detach()
            .abs()
            .to(
                device=self.first.weight.device,
                dtype=self.first.weight.dtype,
            )
        )

        # Case 1: single feature vector
        if x.ndim == 1:
            # (in_features,) -> (1, 1, in_features)
            x = x.unsqueeze(0).unsqueeze(0)

        # Case 2: (batch, in_features)
        elif x.ndim == 2:
            # treat as batch with seq len 1
            x = x.unsqueeze(1)  # (batch, 1, in_features)

        # Case 3+: collapse all middle dims into sequence dimension
        else:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1, x.shape[-1])  # (batch, seq, in_features)

        # first.weight: (hidden_dim, in_features)
        # want per-sample, per-sequence, per-neuron contributions:
        # (batch, seq, hidden_dim)
        scores = torch.matmul(x, self.first.weight.T).abs()

        # seq aggregation = mean
        scores = scores.mean(dim=1)  # (batch, hidden_dim)

        # batch aggregation = L2
        scores = torch.norm(scores, p=2, dim=0)  # (hidden_dim,)

        if self.score_sum is None:
            self.score_sum = scores.clone()
        else:
            self.score_sum += scores

        self.num_updates += 1

    def mean(self) -> torch.Tensor:
        if self.score_sum is None or self.num_updates == 0:
            raise RuntimeError("No activations collected.")
        return self.score_sum / self.num_updates


@torch.no_grad()
def collect_conette_encoder_activation_scores(
    model: nn.Module,
    loader,
    num_batches: int = 10,
    task: str = _UNSET,
) -> dict[str, torch.Tensor]:
    """
    Collect input activation statistics for CoNeTTE encoder pwconv1 layers.
    """
    if task is _UNSET:
        task = config.dataset
    model.eval()

    collectors = {}
    handles = []

    for stage_idx, stage in enumerate(model.preprocessor.encoder.stages):
        for block_idx, block in enumerate(stage):
            if not hasattr(block, "pwconv1"):
                continue
            if not isinstance(block.pwconv1, nn.Linear):
                continue

            name = f"preprocessor.encoder.stages.{stage_idx}.{block_idx}.pwconv1"
            collector = ActivationCollector(block.pwconv1)
            collectors[name] = collector

            def make_hook(c):
                def hook_fn(module, inputs, output):
                    x = inputs[0]
                    c.update(x)

                return hook_fn

            handles.append(block.pwconv1.register_forward_hook(make_hook(collector)))
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        audio = batch["audio"]
        sr = batch["sr"]
        _ = model(audio, sr, task=task)

    for handle in handles:
        handle.remove()

    return {name: collector.mean() for name, collector in collectors.items()}


@torch.no_grad()
def prune_conette(
    model: CoNeTTEModel,
    decoder_threshold: Optional[float] = _UNSET,
    convnext_3072_threshold: Optional[float] = _UNSET,
    convnext_1536_threshold: Optional[float] = _UNSET,
    score_mode: str = _UNSET,
    num_calibration_batches: int = _UNSET,
    verbose: bool = True,
    loader: torch.utils.data.DataLoader = None,
):
    """
    Prune CoNeTTE MLP blocks:
      - decoder: linear1 / linear2
      - encoder: pwconv1 / pwconv2

    Set a threshold to None to skip that part.
    """
    if decoder_threshold is _UNSET:
        decoder_threshold = config.decoder_threshold
    if convnext_3072_threshold is _UNSET:
        convnext_3072_threshold = config.convnext_3072_threshold
    if convnext_1536_threshold is _UNSET:
        convnext_1536_threshold = config.convnext_1536_threshold
    if score_mode is _UNSET:
        score_mode = config.pruning_score_mode
    if num_calibration_batches is _UNSET:
        num_calibration_batches = config.num_calibration_batches
    print(
        f"Pruning with score mode: {score_mode}, decoder_threshold={decoder_threshold}, convnext_3072_threshold={convnext_3072_threshold}, convnext_1536_threshold={convnext_1536_threshold}  "
    )
    model.eval()

    pruned_layer_names = set()

    activation_scores = None
    if loader is not None and score_mode == "wanda":
        activation_scores = collect_conette_encoder_activation_scores(
            model,
            loader=loader,
            num_batches=num_calibration_batches,
        )

    # -------------------------
    # 1) Decoder pruning
    # -------------------------
    if decoder_threshold is not None:
        dec = model.model.decoder
        for li, layer in enumerate(dec.layers):
            old_hidden = layer.linear1.out_features

            new_fc1, new_fc2 = prune_linear_pair(
                layer.linear1,
                layer.linear2,
                threshold=decoder_threshold,
                score_mode="sum_l2",
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
    if convnext_3072_threshold is not None or convnext_1536_threshold is not None:
        for stage_idx, stage in enumerate(model.preprocessor.encoder.stages):
            for block_idx, block in enumerate(stage):
                if not (hasattr(block, "pwconv1") and hasattr(block, "pwconv2")):
                    continue

                pw1 = block.pwconv1
                pw2 = block.pwconv2

                if not isinstance(pw1, nn.Linear) or not isinstance(pw2, nn.Linear):
                    continue

                old_hidden = pw1.out_features
                threshold = None

                if old_hidden == 1536 and convnext_1536_threshold is not None:
                    threshold = convnext_1536_threshold
                elif old_hidden == 3072 and convnext_3072_threshold is not None:
                    threshold = convnext_3072_threshold

                if threshold is None:
                    continue

                base_name = f"preprocessor.encoder.stages.{stage_idx}.{block_idx}"
                block_activation_scores = None

                if activation_scores is not None:
                    act_name = f"{base_name}.pwconv1"
                    if act_name in activation_scores:
                        block_activation_scores = activation_scores[act_name].to(
                            device=pw1.weight.device,
                            dtype=pw1.weight.dtype,
                        )

                new_pw1, new_pw2 = prune_linear_pair(
                    pw1,
                    pw2,
                    threshold=threshold,
                    score_mode=score_mode,
                    activation_scores=block_activation_scores,
                )

                block.pwconv1 = new_pw1
                block.pwconv2 = new_pw2

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
    first: Conv1D, second: Conv1D, threshold: float, score_mode: str = "sum_l2"
):
    d_in, d_hidden = first.weight.shape
    d_hidden_2, d_out = second.weight.shape

    if d_hidden_2 != d_hidden:
        raise RuntimeError(
            f"Incompatible Conv1D pair: first hidden={d_hidden}, second hidden={d_hidden_2}"
        )

    k = int(round(d_hidden * threshold))
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
    gpt_threshold: Optional[float] = _UNSET,
    mapper_threshold: Optional[float] = _UNSET,
    htsat_threshold: Optional[float] = _UNSET,
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
    if gpt_threshold is _UNSET:
        gpt_threshold = config.gpt_threshold
    if mapper_threshold is _UNSET:
        mapper_threshold = config.mapper_threshold
    if htsat_threshold is _UNSET:
        htsat_threshold = config.htsat_threshold
    if htsat_min_hidden_dim is _UNSET:
        htsat_min_hidden_dim = config.htsat_min_hidden_dim

    model.eval()

    # -------------------------
    # 1) GPT2 pruning
    # -------------------------
    if gpt_threshold is not None:
        gpt_blocks = model.gpt.transformer.h

        for i, block in enumerate(gpt_blocks):
            old_hidden = block.mlp.c_fc.weight.shape[1]

            new_c_fc, new_c_proj = prune_conv1d_pair(
                block.mlp.c_fc,
                block.mlp.c_proj,
                threshold=gpt_threshold,
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
    if mapper_threshold is not None:
        mapper_layers = model.clap_project.transformer.layers

        for i, layer in enumerate(mapper_layers):
            old_hidden = layer.mlp.fc1.out_features

            new_fc1, new_fc2 = prune_linear_pair(
                layer.mlp.fc1,
                layer.mlp.fc2,
                threshold=mapper_threshold,
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
    if htsat_threshold is not None:
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
                    threshold=htsat_threshold,
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
