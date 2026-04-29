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

    For mode="wanda", activation_scores must already contain one aggregated
    importance score per hidden neuron, i.e. shape (hidden_dim,).
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
) -> tuple[nn.Linear, nn.Linear]:
    """
    Local pruning of an MLP pair: keep top-`threshold` fraction of hidden neurons
    ranked by their within-layer importance score.

      first : Linear(d_in, d_hidden)
      second: Linear(d_hidden, d_out)
    """
    d_hidden = first.out_features
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
    k = max(1, min(int(round(d_hidden * threshold)), d_hidden))
    keep_idx = torch.topk(scores, k=k, largest=True).indices
    return apply_linear_pair_pruning(first, second, keep_idx)


@torch.no_grad()
def apply_linear_pair_pruning(
    first: nn.Linear,
    second: nn.Linear,
    keep_idx: torch.Tensor,
) -> tuple[nn.Linear, nn.Linear]:
    """
    Apply structured pruning to an MLP pair given explicit keep indices.
    Separated from score computation so global pruning can supply indices directly.
    """
    k = len(keep_idx)
    keep_idx, _ = torch.sort(keep_idx)
    d_in, d_out = first.in_features, second.out_features

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


def allocate_layerwise_keep_indices(
    layer_infos: list[dict],
    global_pruning_ratio: float,
) -> list[torch.Tensor]:
    """
    Global pruning with cost-weighted greedy neuron selection.

    Problem with naive global ranking across heterogeneous layers:
    CoNeTTE has hidden dims 384 / 768 / 1536 / 3072. A 3072-neuron costs ~8×
    more parameters than a 384-neuron. A simple normalized-score threshold
    ignores this: cheap unimportant neurons in small layers survive while
    expensive neurons in large layers are removed — the 3072 layers end up
    under-pruned even at high global ratios.

    Solution — cost-weighted value/cost selection:
      value(neuron) = raw importance score (preserves cross-layer scale information)
      cost(neuron)  = neuron_cost (params removed if this neuron is pruned)

    Neurons are sorted ascending by value/cost. We greedily remove the
    least-value-per-parameter neurons until the target parameter removal is met.
    A 3072-neuron must be ~8× more important per activation unit than a 384-neuron
    to justify keeping it — correctly reflecting its parameter cost.

    Each entry in layer_infos must contain:
      - "scores":      1-D importance tensor (higher = more important)
      - "neuron_cost": parameters removed when one hidden neuron is pruned
      - "num_neurons": hidden dimension of this layer

    Returns:
        keep_indices: list of keep-index tensors, one per layer (same order)
    """
    if not (0.0 <= global_pruning_ratio < 1.0):
        raise ValueError(
            f"global_pruning_ratio must be in [0, 1), got {global_pruning_ratio}"
        )

    total_params = sum(
        info["num_neurons"] * info["neuron_cost"] for info in layer_infos
    )
    target_remove_params = int(round(total_params * global_pruning_ratio))

    # Build flat list of all neurons: (value_per_cost, layer_idx, neuron_idx)
    all_neurons = []
    for li, info in enumerate(layer_infos):
        scores = info["scores"]
        cost = info["neuron_cost"]
        for ni in range(len(scores)):
            all_neurons.append((scores[ni].item() / cost, li, ni))

    # Sort ascending: least value-per-cost first → prune these
    all_neurons.sort(key=lambda x: x[0])

    # Greedy removal until parameter budget is met
    remove_mask = {
        li: torch.zeros(info["num_neurons"], dtype=torch.bool)
        for li, info in enumerate(layer_infos)
    }
    removed_params = 0
    for value_per_cost, li, ni in all_neurons:
        if removed_params >= target_remove_params:
            break
        cost = layer_infos[li]["neuron_cost"]
        if removed_params + cost <= target_remove_params:
            remove_mask[li][ni] = True
            removed_params += cost

    # Convert remove masks to keep indices; guarantee at least 1 neuron per layer
    keep_indices = []
    for li, info in enumerate(layer_infos):
        keep_idx = (~remove_mask[li]).nonzero(as_tuple=True)[0]
        if len(keep_idx) == 0:
            keep_idx = info["scores"].argmax().unsqueeze(0)
        keep_indices.append(keep_idx)

    return keep_indices


def get_conette_hidden_dims(model: CoNeTTEModel) -> dict[str, int]:
    """
    Read the current hidden dimensions from all prunable CoNeTTE layers.

    Returns a dict mapping layer keys to their current hidden dim:
      "preprocessor.encoder.stages.{s}.{b}" -> pwconv1.out_features
      "model.decoder.layers.{i}"             -> linear1.out_features
    """
    dims = {}
    for stage_idx, stage in enumerate(model.preprocessor.encoder.stages):
        for block_idx, block in enumerate(stage):
            if hasattr(block, "pwconv1") and isinstance(block.pwconv1, nn.Linear):
                key = f"preprocessor.encoder.stages.{stage_idx}.{block_idx}"
                dims[key] = block.pwconv1.out_features
    for li, layer in enumerate(model.model.decoder.layers):
        if hasattr(layer, "linear1") and isinstance(layer.linear1, nn.Linear):
            key = f"model.decoder.layers.{li}"
            dims[key] = layer.linear1.out_features
    return dims


@torch.no_grad()
def rebuild_conette_from_dims(
    model: CoNeTTEModel,
    hidden_dims: dict[str, int],
    verbose: bool = False,
) -> CoNeTTEModel:
    """
    Reshape CoNeTTE Linear layers to saved hidden dims WITHOUT copying weights.

    Weights will be loaded from a state dict afterward. This is used when loading
    a KD checkpoint to avoid re-running Wanda (which is non-deterministic due to
    calibration sample ordering → different keep indices → shape mismatch).

    Starts from the full baseline model; replaces each pwconv pair / decoder FC
    pair with a new Linear layer of the saved size.
    """
    for stage_idx, stage in enumerate(model.preprocessor.encoder.stages):
        for block_idx, block in enumerate(stage):
            if not (hasattr(block, "pwconv1") and isinstance(block.pwconv1, nn.Linear)):
                continue
            key = f"preprocessor.encoder.stages.{stage_idx}.{block_idx}"
            if key not in hidden_dims:
                continue
            target_k = hidden_dims[key]
            d_in = block.pwconv1.in_features
            d_out = block.pwconv2.out_features
            has_bias1 = block.pwconv1.bias is not None
            has_bias2 = block.pwconv2.bias is not None
            dev, dt = block.pwconv1.weight.device, block.pwconv1.weight.dtype
            block.pwconv1 = nn.Linear(d_in, target_k, bias=has_bias1).to(
                device=dev, dtype=dt
            )
            block.pwconv2 = nn.Linear(target_k, d_out, bias=has_bias2).to(
                device=dev, dtype=dt
            )
            if verbose:
                print(f"[Rebuild] {key}: hidden dim set to {target_k}")

    for li, layer in enumerate(model.model.decoder.layers):
        key = f"model.decoder.layers.{li}"
        if key not in hidden_dims:
            continue
        if not (hasattr(layer, "linear1") and isinstance(layer.linear1, nn.Linear)):
            continue
        target_k = hidden_dims[key]
        d_in = layer.linear1.in_features
        d_out = layer.linear2.out_features
        has_bias1 = layer.linear1.bias is not None
        has_bias2 = layer.linear2.bias is not None
        dev, dt = layer.linear1.weight.device, layer.linear1.weight.dtype
        layer.linear1 = nn.Linear(d_in, target_k, bias=has_bias1).to(
            device=dev, dtype=dt
        )
        layer.linear2 = nn.Linear(target_k, d_out, bias=has_bias2).to(
            device=dev, dtype=dt
        )
        if verbose:
            print(f"[Rebuild] {key}: hidden dim set to {target_k}")

    return model


class ActivationCollector:
    """
    Collect Minitron-style activation norms for one FFN hidden dimension.

    Hook this on the INPUT to the DOWN-projection (pwconv2 / linear2) to capture
    post-GELU hidden-neuron magnitudes. Multiply result() by the DOWN-projection's
    column norms (||W2_i||) to get the full Minitron importance score.

    Aggregation:
      - sequence/time dimension: mean (PAD positions excluded via sequence_mask)
      - batch dimension: L2 across the full calibration set
    """

    def __init__(self):
        self.score_sumsq = None
        self.num_samples = 0
        self.sequence_mask = None

    def set_sequence_mask(self, mask: Optional[torch.Tensor]) -> None:
        self.sequence_mask = mask

    def clear_sequence_mask(self) -> None:
        self.sequence_mask = None

    def update(self, tensor: torch.Tensor) -> None:
        """
        tensor shape:
          - (hidden_dim,)
          - (batch, hidden_dim)
          - (batch, seq, hidden_dim)
          - or any (..., hidden_dim)

        tensor is the post-GELU for the encoder and the post-ReLU for the decoder input to the DOWN-projection (linear2 / pwconv2).
        """
        x = tensor.detach().abs().float()
        mask = self.sequence_mask

        # Case 1: single feature vector
        if x.ndim == 1:
            # (hidden_dim,) -> (1, 1, hidden_dim)
            x = x.unsqueeze(0).unsqueeze(0)
            if mask is not None:
                mask = mask.reshape(1, -1)

        # Case 2: (batch, hidden_dim)
        elif x.ndim == 2:
            # treat as batch with seq len 1
            x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
            if mask is not None:
                mask = mask.reshape(x.shape[0], -1)

        # Case 3+: collapse all middle dims into sequence dimension
        else:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1, x.shape[-1])  # (batch, seq, hidden_dim)
            if mask is not None:
                mask = mask.reshape(batch_size, -1)

        # x is (batch, seq, hidden_dim) — post-GELU activations; no matmul needed
        scores = x

        if mask is not None:
            mask = mask.to(device=scores.device, dtype=torch.bool)
            if mask.shape != scores.shape[:2]:
                raise RuntimeError(
                    f"Sequence mask shape {mask.shape} does not match scores "
                    f"shape {scores.shape[:2]}"
                )
            scores = scores.masked_fill(~mask.unsqueeze(-1), 0.0)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=scores.dtype)
            scores = scores.sum(dim=1) / denom
        else:
            # seq aggregation = mean(abs)
            scores = scores.mean(dim=1)  # (batch, hidden_dim)

        # batch aggregation = L2 across the whole calibration set, not per-update mean.
        batch_sumsq = scores.pow(2).sum(dim=0)  # (hidden_dim,)
        if self.score_sumsq is None:
            self.score_sumsq = batch_sumsq.clone()
        else:
            self.score_sumsq += batch_sumsq
        self.num_samples += scores.shape[0]

    def result(self) -> torch.Tensor:
        if self.score_sumsq is None or self.num_samples == 0:
            raise RuntimeError("No activations collected.")
        return torch.sqrt(self.score_sumsq)


@torch.no_grad()
def collect_conette_encoder_activation_scores(
    model: nn.Module,
    loader,
    num_batches: int = 10,
    task: str = _UNSET,
) -> dict[str, torch.Tensor]:
    """
    Collect Minitron-style importance scores for CoNeTTE encoder pwconv pairs.

    Hooks the INPUT to pwconv2 (post-GELU activations), then multiplies by
    ||pwconv2.weight[:, i]||_2 per neuron — matching Minitron's formula:
      score_i = ||W_down_i||_2 * L2_calibration(mean_seq(act_i))
    """
    if task is _UNSET:
        task = config.dataset
    model.eval()

    # name -> (ActivationCollector, down_proj)
    collectors: dict[str, tuple] = {}
    handles = []

    for stage_idx, stage in enumerate(model.preprocessor.encoder.stages):
        for block_idx, block in enumerate(stage):
            if not hasattr(block, "pwconv1"):
                continue
            if not isinstance(block.pwconv1, nn.Linear):
                continue

            name = f"preprocessor.encoder.stages.{stage_idx}.{block_idx}.pwconv1"
            collector = ActivationCollector()
            collectors[name] = (collector, block.pwconv2)

            def make_hook(c):
                def hook_fn(module, inputs, output):
                    c.update(inputs[0])

                return hook_fn

            handles.append(block.pwconv2.register_forward_hook(make_hook(collector)))

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        audio = batch["audio"]
        sr = batch["sr"]
        _ = model(audio, sr, task=task)

    for handle in handles:
        handle.remove()

    result = {}
    for name, (collector, down_proj) in collectors.items():
        result[name] = collector.result()
    return result


@torch.no_grad()
def collect_conette_decoder_activation_scores(
    model: CoNeTTEModel,
    loader,
    num_batches: int = 10,
    dataset_name: str = _UNSET,
) -> dict[str, torch.Tensor]:
    """
    Collect input activation statistics for CoNeTTE decoder linear1 layers
    using teacher-forced forward passes.

    Teacher forcing is critical: beam-search inference activates decoder neurons
    with generated tokens (BOS-dominated), not gold captions. This gives misleading
    importance scores because the activation distribution at inference time differs
    fundamentally from training. Teacher forcing matches the training distribution.
    """
    # Local import to avoid circular dependency at module level
    from finetune import prepare_batch

    if dataset_name is _UNSET:
        dataset_name = config.dataset
    model.eval()
    pad_idx = model.model.tokenizer.pad_token_id

    # name -> (ActivationCollector, down_proj)
    collectors: dict[str, tuple] = {}
    handles = []

    for li, layer in enumerate(model.model.decoder.layers):
        if not hasattr(layer, "linear1") or not isinstance(layer.linear1, nn.Linear):
            continue
        name = f"model.decoder.layers.{li}.linear1"
        collector = ActivationCollector()
        collectors[name] = (collector, layer.linear2)

        def make_hook(c):
            def hook_fn(module, inputs, output):
                x = inputs[0]
                # CoNeTTE's AACTransformerDecoder uses batch_first=False, so
                # linear2 sees (tgt_len, batch, hidden_dim). Transpose to
                # (batch, tgt_len, hidden_dim) before aggregation; otherwise
                # the collector swaps sequence and batch semantics and ends up
                # computing an L2 over tokens instead of over samples.
                if x.ndim == 3:
                    x = x.transpose(0, 1)
                c.update(x)

            return hook_fn

        handles.append(layer.linear2.register_forward_hook(make_hook(collector)))

    for batch_idx, raw_batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        batch = prepare_batch(model, raw_batch, dataset_name)
        audio = batch["audio"]
        audio_shape = batch["audio_shape"]
        caps_in = batch["captions"][:, :-1]
        token_mask = caps_in != pad_idx
        for collector, _ in collectors.values():
            collector.set_sequence_mask(token_mask)
        encoder_outs = model.model.encode_audio(audio, audio_shape)
        try:
            _ = model.model.decode_audio(encoder_outs, "forcing", caps_in=caps_in)
        finally:
            for collector, _ in collectors.values():
                collector.clear_sequence_mask()

    for handle in handles:
        handle.remove()

    result = {}
    for name, (collector, _) in collectors.items():
        result[name] = collector.result()
    return result


@torch.no_grad()
def prune_conette(
    model: CoNeTTEModel,
    decoder_pruning_ratio: Optional[float] = _UNSET,
    convnext_3072_threshold: Optional[float] = _UNSET,
    convnext_1536_threshold: Optional[float] = _UNSET,
    global_pruning_ratio: Optional[float] = _UNSET,
    score_mode: str = _UNSET,
    num_calibration_batches: int = _UNSET,
    verbose: bool = True,
    loader: torch.utils.data.DataLoader = None,
):
    """
    Prune CoNeTTE MLP blocks (encoder pwconv pairs + decoder linear pairs).

    decoder_pruning_ratio: fraction of decoder neurons to REMOVE (None = skip decoder).
    global_pruning_ratio:  fraction of encoder neurons to REMOVE globally.
    convnext_*_threshold:  per-layer local keep-ratio for encoder (None = skip).
    """
    if decoder_pruning_ratio is _UNSET:
        decoder_pruning_ratio = config.decoder_pruning_ratio
    if convnext_3072_threshold is _UNSET:
        convnext_3072_threshold = config.convnext_3072_threshold
    if convnext_1536_threshold is _UNSET:
        convnext_1536_threshold = config.convnext_1536_threshold
    if global_pruning_ratio is _UNSET:
        global_pruning_ratio = getattr(config, "global_pruning_ratio", None)
    if score_mode is _UNSET:
        score_mode = config.pruning_score_mode
    if num_calibration_batches is _UNSET:
        num_calibration_batches = config.num_calibration_batches

    model.eval()
    pruned_layer_names = set()

    encoder_activation_scores = None
    decoder_activation_scores = None
    if loader is not None and score_mode == "wanda":
        if (
            global_pruning_ratio is not None
            or convnext_3072_threshold is not None
            or convnext_1536_threshold is not None
        ):
            encoder_activation_scores = collect_conette_encoder_activation_scores(
                model,
                loader=loader,
                num_batches=num_calibration_batches,
            )
        if decoder_pruning_ratio is not None:
            decoder_activation_scores = collect_conette_decoder_activation_scores(
                model,
                loader=loader,
                num_batches=num_calibration_batches,
                dataset_name=config.dataset,
            )

    # -------------------------
    # 1) Decoder pruning
    # -------------------------
    if decoder_pruning_ratio is not None:
        dec = model.model.decoder
        print(
            f"[Decoder] Global pruning | ratio={decoder_pruning_ratio} | score_mode={score_mode}"
        )

        layer_scores = []
        for li, layer in enumerate(dec.layers):
            act = None
            if decoder_activation_scores is not None:
                key = f"model.decoder.layers.{li}.linear1"
                act = decoder_activation_scores.get(key)
            if act is not None:
                act = act.to(
                    device=layer.linear1.weight.device, dtype=layer.linear1.weight.dtype
                )
            scores = compute_linear_hidden_scores(
                layer.linear1, layer.linear2, mode=score_mode, activation_scores=act
            )
            layer_scores.append(scores)

        # Global keep/remove: rank all neurons jointly, prune bottom fraction
        all_scores = torch.cat(layer_scores)
        n_total = len(all_scores)
        n_keep = max(
            len(dec.layers), n_total - int(round(n_total * decoder_pruning_ratio))
        )
        keep_mask = torch.zeros(n_total, dtype=torch.bool)
        keep_mask[torch.topk(all_scores, k=n_keep, largest=True).indices] = True

        offsets = torch.tensor([0] + [len(s) for s in layer_scores]).cumsum(0)
        for li, layer in enumerate(dec.layers):
            old_hidden = layer.linear1.out_features
            keep_idx = keep_mask[offsets[li] : offsets[li + 1]].nonzero(as_tuple=True)[
                0
            ]
            if len(keep_idx) == 0:
                keep_idx = layer_scores[li].argmax().unsqueeze(0)
            new_fc1, new_fc2 = apply_linear_pair_pruning(
                layer.linear1, layer.linear2, keep_idx
            )
            layer.linear1 = new_fc1
            layer.linear2 = new_fc2
            pruned_layer_names.add(f"model.decoder.layers.{li}.linear1")
            pruned_layer_names.add(f"model.decoder.layers.{li}.linear2")
            if verbose:
                print(
                    f"[Decoder layer {li}] hidden dim: {old_hidden} -> {new_fc1.out_features} "
                    f"(kept {len(keep_idx)}/{old_hidden})"
                )

    # -------------------------
    # 2) Encoder pruning
    # -------------------------
    # Collect eligible encoder blocks
    enc_blocks = []  # (stage_idx, block_idx, block, base_name)
    for stage_idx, stage in enumerate(model.preprocessor.encoder.stages):
        for block_idx, block in enumerate(stage):
            if not (hasattr(block, "pwconv1") and hasattr(block, "pwconv2")):
                continue
            if not isinstance(block.pwconv1, nn.Linear) or not isinstance(
                block.pwconv2, nn.Linear
            ):
                continue
            enc_blocks.append(
                (
                    stage_idx,
                    block_idx,
                    block,
                    f"preprocessor.encoder.stages.{stage_idx}.{block_idx}",
                )
            )

    if global_pruning_ratio is not None:
        # ── Global mode: layer-aware budget allocation + local top-k ─────────
        print(
            f"[Encoder] Global pruning | ratio={global_pruning_ratio} | score_mode={score_mode}"
        )

        layer_infos = []
        for stage_idx, block_idx, block, base_name in enc_blocks:
            act = (
                encoder_activation_scores.get(f"{base_name}.pwconv1")
                if encoder_activation_scores
                else None
            )
            if act is not None:
                act = act.to(
                    device=block.pwconv1.weight.device,
                    dtype=block.pwconv1.weight.dtype,
                )

            scores = compute_linear_hidden_scores(
                block.pwconv1,
                block.pwconv2,
                mode=score_mode,
                activation_scores=act,
            )

            neuron_cost = (
                block.pwconv1.in_features
                + block.pwconv2.out_features
                + int(block.pwconv1.bias is not None)
                + int(block.pwconv2.bias is not None)
            )

            layer_infos.append(
                {
                    "stage_idx": stage_idx,
                    "block_idx": block_idx,
                    "block": block,
                    "base_name": base_name,
                    "scores": scores,
                    "num_neurons": block.pwconv1.out_features,
                    "neuron_cost": neuron_cost,
                }
            )

        keep_indices = allocate_layerwise_keep_indices(
            layer_infos, global_pruning_ratio
        )

        for info, keep_idx in zip(layer_infos, keep_indices):
            stage_idx = info["stage_idx"]
            block_idx = info["block_idx"]
            block = info["block"]
            base_name = info["base_name"]
            old_hidden = info["num_neurons"]

            new_pw1, new_pw2 = apply_linear_pair_pruning(
                block.pwconv1,
                block.pwconv2,
                keep_idx,
            )

            block.pwconv1 = new_pw1
            block.pwconv2 = new_pw2

            pruned_layer_names.add(f"{base_name}.pwconv1")
            pruned_layer_names.add(f"{base_name}.pwconv2")

            if verbose:
                k = len(keep_idx)
                print(
                    f"[ConvNeXt stage {stage_idx} block {block_idx}] "
                    f"hidden dim: {old_hidden} -> {new_pw1.out_features} "
                    f"(kept {k}/{old_hidden}, neuron_cost={info['neuron_cost']})"
                )

    elif convnext_3072_threshold is not None or convnext_1536_threshold is not None:
        # ── Local mode: same threshold per layer type ─────────────────────────
        print(
            f"[Encoder] Local pruning | 3072_threshold={convnext_3072_threshold} "
            f"1536_threshold={convnext_1536_threshold} | score_mode={score_mode}"
        )
        for stage_idx, block_idx, block, base_name in enc_blocks:
            old_hidden = block.pwconv1.out_features
            if old_hidden == 1536 and convnext_1536_threshold is not None:
                threshold = convnext_1536_threshold
            elif old_hidden == 3072 and convnext_3072_threshold is not None:
                threshold = convnext_3072_threshold
            else:
                continue

            act = (
                encoder_activation_scores.get(f"{base_name}.pwconv1")
                if encoder_activation_scores
                else None
            )
            if act is not None:
                act = act.to(
                    device=block.pwconv1.weight.device, dtype=block.pwconv1.weight.dtype
                )

            new_pw1, new_pw2 = prune_linear_pair(
                block.pwconv1,
                block.pwconv2,
                threshold=threshold,
                score_mode=score_mode,
                activation_scores=act,
            )
            block.pwconv1 = new_pw1
            block.pwconv2 = new_pw2
            pruned_layer_names.add(f"{base_name}.pwconv1")
            pruned_layer_names.add(f"{base_name}.pwconv2")
            if verbose:
                print(
                    f"[ConvNeXt stage {stage_idx} block {block_idx}] "
                    f"hidden dim: {old_hidden} -> {new_pw1.out_features}"
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
@torch.no_grad()
def collect_clapcap_activation_scores(
    model,
    audio_paths: list,
    num_samples: int = 128,
) -> dict[str, torch.Tensor]:
    """
    Collect Minitron-style importance scores for CLAPCAP HTSAT and Mapper MLP pairs.

    Hooks the INPUT to fc2 (post-GELU activations from fc1) in every HTSAT block
    and every Mapper transformer layer. Runs audio through the encoder and mapper
    (no GPT-2 generation — hooks fire before generation begins).

    Aggregation: seq=mean, batch=L2 (same formula as CoNeTTE encoder/decoder).
    Returns dict mapping layer key -> score tensor of shape (hidden_dim,).
    Each score is: L2_calibration(mean_seq(act_i)) × ||fc2.weight[:, i]||_2
    """
    clapcap = model.clapcap
    clapcap.eval()

    collectors: dict[str, tuple] = {}
    handles = []

    for stage_idx, stage in enumerate(clapcap.clap.base.htsat.layers):
        if not hasattr(stage, "blocks"):
            continue
        for block_idx, block in enumerate(stage.blocks):
            if not hasattr(block, "mlp"):
                continue
            key = f"htsat.stage{stage_idx}.block{block_idx}"
            collector = ActivationCollector()
            collectors[key] = (collector, block.mlp.fc2)

            def make_hook(c):
                def hook_fn(module, inputs, output):
                    c.update(inputs[0])

                return hook_fn

            handles.append(block.mlp.fc2.register_forward_hook(make_hook(collector)))

    for i, layer in enumerate(clapcap.clap_project.transformer.layers):
        key = f"mapper.layer{i}"
        collector = ActivationCollector()
        collectors[key] = (collector, layer.mlp.fc2)

        def make_hook(c):
            def hook_fn(module, inputs, output):
                c.update(inputs[0])

            return hook_fn

        handles.append(layer.mlp.fc2.register_forward_hook(make_hook(collector)))

    n_collected = 0
    for path in audio_paths:
        if n_collected >= num_samples:
            break
        try:
            audio = model.preprocess_audio([path], resample=True).squeeze(1)
            if torch.cuda.is_available():
                audio = audio.cuda()
            prefix, _ = clapcap.clap(audio)
            if model.args.normalize_prefix:
                prefix = prefix / prefix.norm(2, -1).reshape(-1, 1)
            clapcap.clap_project(prefix)
            n_collected += 1
            if n_collected % 20 == 0:
                print(
                    f"[CLAPCAP calibration] {n_collected}/{num_samples} samples processed"
                )
        except Exception as e:
            print(f"[CLAPCAP calibration] skipping {path}: {e}")

    for handle in handles:
        handle.remove()

    print(f"[CLAPCAP calibration] done — {n_collected} samples collected")

    result = {}
    for key, (collector, _) in collectors.items():
        result[key] = collector.result()
    return result


def prune_clapcap(
    model,
    htsat_pruning_ratio: Optional[float] = _UNSET,
    mapper_pruning_ratio: Optional[float] = _UNSET,
    score_mode: str = _UNSET,
    num_calibration_batches: int = _UNSET,
    audio_paths: Optional[list] = None,
    verbose: bool = True,
):
    """
    Structured pruning for CLAPCAP.

    Components pruned:
      - HTSAT encoder MLP layers (global ratio across all HTSAT stages)
      - Mapper transformer MLP layers (global ratio across all mapper layers)

    GPT-2 decoder: EXCLUDED. Pruning Conv1D pairs in GPT-2 caused a 6× inference
    time increase (likely due to non-contiguous weight layout after in-place slicing).
    The decoder is 227M params but dominates inference time — pruning it is not
    viable without architectural changes.

    Global pruning within each component uses per-layer max-normalization before
    ranking, removing cross-layer scale bias (same as CoNeTTE encoder pruning).
    """
    if htsat_pruning_ratio is _UNSET:
        htsat_pruning_ratio = config.htsat_pruning_ratio
    if mapper_pruning_ratio is _UNSET:
        mapper_pruning_ratio = config.mapper_pruning_ratio
    if score_mode is _UNSET:
        score_mode = config.pruning_score_mode
    if num_calibration_batches is _UNSET:
        num_calibration_batches = config.num_calibration_batches

    clapcap = model.clapcap
    clapcap.eval()

    htsat_scores: dict[str, torch.Tensor] = {}
    mapper_scores: dict[str, torch.Tensor] = {}
    if score_mode == "wanda" and audio_paths:
        all_scores = collect_clapcap_activation_scores(
            model, audio_paths=audio_paths, num_samples=num_calibration_batches
        )
        htsat_scores = {k: v for k, v in all_scores.items() if k.startswith("htsat")}
        mapper_scores = {k: v for k, v in all_scores.items() if k.startswith("mapper")}

    # ── HTSAT global pruning ───────────────────────────────────────────────────
    if htsat_pruning_ratio is not None:
        layer_infos = []

        for stage_idx, stage in enumerate(clapcap.clap.base.htsat.layers):
            if not hasattr(stage, "blocks"):
                continue
            for block_idx, block in enumerate(stage.blocks):
                if not hasattr(block, "mlp"):
                    continue
                key = f"htsat.stage{stage_idx}.block{block_idx}"
                fc1, fc2 = block.mlp.fc1, block.mlp.fc2
                act = htsat_scores.get(key)
                if act is not None:
                    act = act.to(device=fc1.weight.device, dtype=fc1.weight.dtype)
                scores = compute_linear_hidden_scores(
                    fc1, fc2, mode=score_mode, activation_scores=act
                )
                neuron_cost = (
                    fc1.in_features
                    + fc2.out_features
                    + int(fc1.bias is not None)
                    + int(fc2.bias is not None)
                )
                layer_infos.append(
                    {
                        "key": key,
                        "mlp": block.mlp,
                        "scores": scores,
                        "num_neurons": fc1.out_features,
                        "neuron_cost": neuron_cost,
                    }
                )

        keep_indices = allocate_layerwise_keep_indices(layer_infos, htsat_pruning_ratio)

        for info, keep_idx in zip(layer_infos, keep_indices):
            old_hidden = info["mlp"].fc1.out_features
            info["mlp"].fc1, info["mlp"].fc2 = apply_linear_pair_pruning(
                info["mlp"].fc1, info["mlp"].fc2, keep_idx
            )
            if verbose:
                print(
                    f"[HTSAT {info['key']}] hidden: {old_hidden} → {info['mlp'].fc1.out_features}"
                )

    # ── Mapper global pruning ──────────────────────────────────────────────────
    if mapper_pruning_ratio is not None:
        layer_scores, layer_mlps = [], []

        for i, layer in enumerate(clapcap.clap_project.transformer.layers):
            key = f"mapper.layer{i}"
            fc1, fc2 = layer.mlp.fc1, layer.mlp.fc2
            act = mapper_scores.get(key)
            if act is not None:
                act = act.to(device=fc1.weight.device, dtype=fc1.weight.dtype)
            scores = compute_linear_hidden_scores(
                fc1, fc2, mode=score_mode, activation_scores=act
            )
            layer_scores.append(scores)
            layer_mlps.append(layer.mlp)

        layer_scores_norm = [s / (s.max() + 1e-12) for s in layer_scores]
        all_cat = torch.cat(layer_scores_norm)
        n_total = len(all_cat)
        n_keep = max(
            len(layer_mlps), n_total - int(round(n_total * mapper_pruning_ratio))
        )
        keep_mask = torch.zeros(n_total, dtype=torch.bool)
        keep_mask[torch.topk(all_cat, k=n_keep, largest=True).indices] = True

        offsets = torch.tensor([0] + [len(s) for s in layer_scores]).cumsum(0)
        for i, mlp in enumerate(layer_mlps):
            keep_idx = keep_mask[offsets[i] : offsets[i + 1]].nonzero(as_tuple=True)[0]
            if len(keep_idx) == 0:
                keep_idx = layer_scores[i].argmax().unsqueeze(0)
            old_hidden = mlp.fc1.out_features
            mlp.fc1, mlp.fc2 = apply_linear_pair_pruning(mlp.fc1, mlp.fc2, keep_idx)
            if verbose:
                print(
                    f"[Mapper layer {i}] hidden: {old_hidden} → {mlp.fc1.out_features}"
                )

    return model
