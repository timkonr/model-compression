import torch
from torch.utils.data import DataLoader
from aac_metrics import evaluate
from aac_datasets import Clotho, AudioCaps
from aac_datasets.utils.collate import BasicCollate
import json
import argparse
from time import perf_counter
from utils.utils import build_samples, load_model
from utils.model_size import (
    get_model_size,
    get_model_params,
    measure_flops_conette,
    measure_flops_clapcap,
)
from utils import config
import datetime
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("--config", help="Path to a YAML experiment config file.")
    parser.add_argument(
        "--path",
        help="Path to a previously saved inference results JSON file.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print some debug output"
    )
    args = parser.parse_args()
    return args


def _technique_name():
    if config.pruning and config.quantization:
        return "pruning+quantization"
    if config.pruning:
        return "pruning"
    if config.quantization:
        return "quantization"
    if config.kd:
        return "kd"
    return "baseline"


def prepare_dataloader(verbose, subset):
    print(f"loading dataset {config.dataset} {subset}")

    if config.dataset == "clotho":

        ds = Clotho(config.data_folder, subset=subset)
    elif config.dataset == "audiocaps":
        ds = AudioCaps(
            config.data_folder,
            subset=subset,
            audio_format="wav",
            sr=22050,
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    if verbose:
        for i in range(len(ds)):
            try:
                print(i, ds[i]["fname"])
            except Exception as e:
                print(f"Exception on index {i}: {str(e)}")

    return DataLoader(ds, batch_size=1, collate_fn=BasicCollate())


def inference(model: torch.nn.Module, data_loader):
    model.eval() if config.baseline_model == "conette" else model.clapcap.eval()
    predictions, references = [], []

    start = perf_counter()
    with torch.no_grad():
        if config.baseline_model == "clapcap":
            csv_path = (
                config.dataset == "clotho"
                and f"{config.data_folder}/CLOTHO_v2.1/clotho_csv_files/clotho_captions_evaluation.csv"
                or f"{config.data_folder}/AUDIOCAPS/csv_files_v1/test.csv"
            )
            audio_path = (
                config.dataset == "clotho"
                and f"{config.data_folder}/CLOTHO_v2.1/clotho_audio_files/evaluation"
                or f"{config.data_folder}/AUDIOCAPS/audio_22050Hz/test"
            )
            samples = build_samples(csv_path)
            start = perf_counter()
            for filename, captions in samples.items():
                path = f"{audio_path}/{filename}"
                if os.path.exists(path):
                    pred = model.generate_caption([path])
                    predictions.extend(pred)
                    references.append(captions)

        elif config.baseline_model == "conette":
            for i, batch in enumerate(data_loader):
                audio = batch["audio"]
                sr = batch["sr"]
                outputs = model(audio, sr, task=config.dataset)
                if i < 1:
                    print("sample model output:", outputs)
                predictions.extend(outputs["cands"])
                references.extend(batch["captions"])

    inference_time = perf_counter() - start
    print(f"inference completed in {inference_time:.3f} seconds")
    return predictions, references, inference_time


def perform_inference(verbose):
    # test_loader: evaluation subset (Clotho="eval", AudioCaps="test") — used for inference
    test_loader = prepare_dataloader(
        verbose, subset="test" if config.dataset == "audiocaps" else "eval"
    )

    # calib_loader: training subset (Clotho="dev", AudioCaps="train") — only needed for
    # activation-aware pruning calibration (score_mode=wanda). Must NOT be test data.
    needs_calib = config.pruning and config.pruning_score_mode == "wanda"
    calib_loader = None
    if needs_calib:
        train_subset = "train" if config.dataset == "audiocaps" else "dev"
        calib_loader = prepare_dataloader(verbose, subset=train_subset)

    model = load_model(
        quantized=config.quantization,
        pruned=config.pruning,
        kd=config.kd,
        loader=calib_loader,
    )

    technique = _technique_name()
    torch_model = model if config.baseline_model == "conette" else model.clapcap
    device = str(next(torch_model.parameters()).device)
    model_size_mb = get_model_size(torch_model)
    model_params = get_model_params(torch_model)

    # FLOPs: skip for quantized-only runs — dynamic quantization changes bit-width but not
    # operation count. FlopCounterMode does not register INT8 aten ops, so measuring a
    # quantized model yields a severe undercount. Model size is the relevant metric for
    # quantization; FLOPs are the relevant metric for pruning/KD.
    print("measuring FLOPs...")
    if config.quantization and not config.pruning:
        print(
            "[FLOPs] Skipped for quantization-only: FLOPs counting not working properly on quantized models. Consider unquantized model instead, as the number of operations does not change anyway."
        )
        flops = None
    elif config.baseline_model == "conette":
        flops = measure_flops_conette(model, test_loader, task=config.dataset)
    else:
        csv_path = (
            f"{config.data_folder}/CLOTHO_v2.1/clotho_csv_files/clotho_captions_evaluation.csv"
            if config.dataset == "clotho"
            else f"{config.data_folder}/AUDIOCAPS/csv_files_v1/test.csv"
        )
        audio_dir = (
            f"{config.data_folder}/CLOTHO_v2.1/clotho_audio_files/evaluation"
            if config.dataset == "clotho"
            else f"{config.data_folder}/AUDIOCAPS/audio_22050Hz/test"
        )
        sample_paths = []
        for filename in build_samples(csv_path):
            candidate = f"{audio_dir}/{filename}"
            if os.path.exists(candidate):
                sample_paths.append(candidate)
            if len(sample_paths) >= 10:
                break
        flops = measure_flops_clapcap(model, sample_paths)

    print(f"starting inference on device: {device}")
    predictions, references, inference_time = inference(model, data_loader=test_loader)

    metadata = {
        "model": config.baseline_model,
        "compression_technique": technique,
        "dataset": config.dataset,
        "seed": config.seed,
        "model_size_mb": model_size_mb,
        "unquantized_parameters": model_params,
        "flops": flops,
        "device": device,
        "inference_time_in_s": f"{inference_time:.3f}",
        "predictions": predictions,
        "references": references,
    }

    if config.pruning:
        if config.baseline_model == "conette":
            metadata["pruning_setup"] = {
                "decoder_threshold": config.decoder_threshold,
                "convnext_3072_threshold": config.convnext_3072_threshold,
                "convnext_1536_threshold": config.convnext_1536_threshold,
                "global_pruning_ratio": getattr(config, "global_pruning_ratio", None),
                "score_mode": config.pruning_score_mode,
                "num_calibration_batches": config.num_calibration_batches,
            }
        elif config.baseline_model == "clapcap":
            metadata["pruning_setup"] = {
                "gpt_threshold": config.gpt_threshold,
                "htsat_threshold": config.htsat_threshold,
                "mapper_threshold": config.mapper_threshold,
                "htsat_min_hidden_dim": config.htsat_min_hidden_dim,
                "score_mode": config.pruning_score_mode,
            }

    return metadata


def load_previous_results(path):
    if not os.path.isfile(path):
        raise ValueError(f"{path} is not a file.")
    with open(path, "r") as fp:
        return json.load(fp)


def save_result(result, fpath):
    os.makedirs("results", exist_ok=True)
    with open(fpath, "w") as fp:
        json.dump(result, fp, indent=2)


def perform_evaluation(inference_result):
    print(f"calculating evaluation metrics...")
    predictions = inference_result.pop("predictions")
    references = inference_result.pop("references")
    corpus_scores, _ = evaluate(
        candidates=predictions,
        mult_references=references,
        metrics=config.metrics,
    )
    inference_result["evaluation_results"] = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in corpus_scores.items()
    }
    print("evaluation completed")
    return inference_result


def main():
    args = parse_args()

    if args.config:
        config.load_from_yaml(args.config)
        config.set_seed(config.seed)

    if not config.inference and not config.evaluation:
        raise ValueError("Doing neither inference nor evaluation")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    if args.path:
        result = load_previous_results(args.path)
    elif config.inference:
        result = perform_inference(args.verbose)
        if config.save_inference_results:
            save_result(
                result,
                f"results/inference_{result['compression_technique']}_{result['dataset']}_{ts}.json",
            )
    else:
        raise ValueError("inference=False but no --path given")

    if config.evaluation:
        result = perform_evaluation(result)
        filename = f"results/eval_{result['compression_technique']}_{result['dataset']}_{ts}.json"
        print(f"saving evaluation results to {filename}")
        save_result(result, filename)


if __name__ == "__main__":
    main()
