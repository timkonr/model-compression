import json

from conette import CoNeTTEConfig, CoNeTTEModel
from prune import prune_clapcap, prune_conette, rebuild_conette_from_dims
from utils import config
from quantize import make_quantized_model
from utils.model_size import get_model_size, get_model_params
from torch.utils.data import DataLoader
from msclap import CLAP
import csv
import os
import torch


def load_model(
    model_path=None,
    quantized=False,
    pruned=False,
    kd=False,
    loader=None,
    verbose=True,
):
    if model_path is None:
        model_path = config.model_folder
    print(f"loading model {config.baseline_model}")
    baseline_path = model_path + "baseline/"

    if config.baseline_model == "conette":
        model = CoNeTTEModel.from_pretrained(
            baseline_path, config=CoNeTTEConfig.from_pretrained(baseline_path)
        )
    elif config.baseline_model == "clapcap":
        model = CLAP(version="clapcap")
    else:
        raise ValueError(f"Unknown baseline model: {config.baseline_model}")

    # model.to("cpu")
    if verbose:
        print(
            f"Original model size on disk: {get_model_size(model if config.baseline_model == "conette" else model.clapcap):.2f} MB"
        )
        print(
            f"Original model params: {get_model_params(model if config.baseline_model == "conette" else model.clapcap)}"
        )
    if kd:
        # KD checkpoint was saved from a pruned model, so its state dict has different
        # tensor shapes than the original config expects. We must:
        #   1. rebuild the pruned architecture (same pruning config used during KD training)
        #   2. load only the state dict from the KD checkpoint — not from_pretrained
        kd_path = config.kd_model

        with open(os.path.join(kd_path, "meta.json"), "r") as f:
            kd_config = json.load(f)
            print(
                f"KD checkpoint was trained with pruning config: {kd_config['pruning']}"
            )

        # Load state dict (always needed — also used to extract dims in legacy path)
        state_dict_path = os.path.join(kd_path, "pytorch_model.bin")
        if not os.path.exists(state_dict_path):
            state_dict_path = os.path.join(kd_path, "model.safetensors")
        if os.path.exists(state_dict_path) and state_dict_path.endswith(".bin"):
            state_dict = torch.load(state_dict_path, map_location="cpu")
        elif os.path.exists(state_dict_path) and state_dict_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(state_dict_path)
        else:
            raise FileNotFoundError(f"No model weights found in {kd_path}")

        hidden_dims = kd_config.get("hidden_dims")
        if not hidden_dims:
            # Legacy: no hidden_dims saved → extract exact shapes from the state dict.
            # pwconv1.weight shape is (hidden_dim, d_in) and linear1.weight is (hidden_dim, d_in).
            print(
                "[KD load] No hidden_dims in meta.json — extracting dims from state dict "
                "(fixes wanda non-determinism for old checkpoints)"
            )
            hidden_dims = {}
            for key, tensor in state_dict.items():
                # Encoder blocks: preprocessor.encoder.stages.S.B.pwconv1.weight
                if key.endswith(".pwconv1.weight"):
                    layer_key = key[: -len(".pwconv1.weight")]
                    hidden_dims[layer_key] = tensor.shape[0]
                # Decoder blocks: model.decoder.layers.L.linear1.weight
                elif key.endswith(".linear1.weight") and "decoder" in key:
                    layer_key = key[: -len(".linear1.weight")]
                    hidden_dims[layer_key] = tensor.shape[0]
            print(f"[KD load] Extracted hidden_dims for {len(hidden_dims)} layers from state dict")

        print(
            f"[KD load] Rebuilding architecture from hidden_dims ({len(hidden_dims)} layers)"
        )
        model = rebuild_conette_from_dims(model, hidden_dims, verbose=True)

        result = model.load_state_dict(state_dict, strict=True)
        if result.missing_keys:
            print(f"[KD load] WARNING: missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"[KD load] WARNING: unexpected keys: {result.unexpected_keys}")
        print(
            f"[KD load] Loaded state dict from {state_dict_path} — {len(state_dict)} keys, no shape mismatches"
        )
    if pruned:
        print("pruning model using setup:")
        if config.baseline_model == "conette":
            print(
                f"decoder_pruning_ratio: {config.decoder_pruning_ratio}, convnext_3072_threshold: {config.convnext_3072_threshold}, convnext_1536_threshold: {config.convnext_1536_threshold}, score_mode: {config.pruning_score_mode}, num_calibration_batches: {config.num_calibration_batches}"
            )
        elif config.baseline_model == "clapcap":
            print(
                f"gpt_threshold: {config.gpt_threshold}, htsat_threshold: {config.htsat_threshold}, mapper_threshold: {config.mapper_threshold}, htsat_min_hidden_dim: {config.htsat_min_hidden_dim}, score_mode: {config.pruning_score_mode}"
            )
        if config.baseline_model == "conette":
            model, pruned_layer_names = prune_conette(
                model, verbose=True, loader=loader
            )
            # finetune_conette(
            #     hf_model=model,
            #     dataset_name=config.dataset,
            #     pruned_layer_names=pruned_layer_names,
            # )
        elif config.baseline_model == "clapcap":
            model.clapcap = prune_clapcap(model.clapcap, verbose=True)
    if quantized:
        model = make_quantized_model(
            model, quantization_mode=config.quantization_mode, loader=loader
        )
    # model.to("cpu")  # ensure model is on CPU
    if verbose:
        new_model_type = (
            "Pruned and quantized"
            if quantized and pruned
            else "Pruned" if pruned else "Quantized" if quantized else "original"
        )
        if new_model_type != "original":
            print(
                f"{new_model_type} model size on disk: {get_model_size(model if config.baseline_model == "conette" else model.clapcap):.2f} MB"
            )
            print(
                f"{new_model_type} unquantized model params: {get_model_params(model if config.baseline_model == "conette" else model.clapcap)}"
            )
    return model


def build_samples(csv_path):
    samples_dict = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if config.dataset == "clotho":
                filename = row["file_name"]
                sample = [
                    row["caption_1"],
                    row["caption_2"],
                    row["caption_3"],
                    row["caption_4"],
                    row["caption_5"],
                ]
                samples_dict[filename] = sample
            elif config.dataset == "audiocaps":
                filename = f"{row['youtube_id']}_{row['start_time']}.wav"
                if filename in samples_dict:
                    samples_dict[filename].append(row["caption"])
                else:
                    samples_dict[filename] = [row["caption"]]

    return samples_dict
