from conette import CoNeTTEConfig, CoNeTTEModel
from prune import prune_clapcap, prune_conette
from utils import config
from quantize import make_quantized_model
from utils.model_size import get_model_size, get_model_params
from torch.utils.data import DataLoader
from msclap import CLAP
import csv


def prepare_models(loader: DataLoader):
    # Load models
    models_to_eval = []
    if config.baseline:
        models_to_eval.append({"model": load_model(), "name": "baseline"})
    if config.quantization:
        models_to_eval.append(
            {"model": load_model(quantized=True, loader=loader), "name": "quantized"}
        )
    if config.pruning:
        models_to_eval.append(
            {"model": load_model(pruned=True, loader=loader), "name": "pruned"}
        )
    if config.kd:
        models_to_eval.append({"model": load_model(kd=True), "name": "kd"})
    return models_to_eval


def prepare_multi_compressed_model(loader: DataLoader):
    return {
        "model": load_model(quantized=True, pruned=True, loader=loader),
        "name": "pruned and quantized",
    }


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
    print("loading model")
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
        kd_path = config.kd_model
        model = CoNeTTEModel.from_pretrained(
            kd_path, config=CoNeTTEConfig.from_pretrained(kd_path)
        )
    if pruned:
        # model = prune(model, keep_ratio=0.5)
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
        print(
            f"{new_model_type} model size on disk: {get_model_size(model if config.baseline_model == "conette" else model.clapcap):.2f} MB"
        )
        print(
            f"{new_model_type} model params: {get_model_params(model if config.baseline_model == "conette" else model.clapcap)}"
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
