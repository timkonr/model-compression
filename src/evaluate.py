import torch
from torch.utils.data import DataLoader
from aac_metrics import evaluate
from aac_datasets import Clotho, AudioCaps
from aac_datasets.utils.collate import BasicCollate
import json
import argparse
from time import perf_counter
from utils.utils import build_samples, get_model_size, get_model_params, prepare_models
from utils import config
import datetime
import os


def parse_args():
    # Parse args
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument(
        "--cpu", action="store_true", default=True, help="Evaluate on CPU"
    )
    parser.add_argument(
        "--gpu",
        dest="cpu",
        action="store_false",
        help="Evaluate on GPU (not available for quantization)",
    )
    parser.add_argument(
        "--path",
        help="Path to folder containing predictions and references. Only used when config.inference is False.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print some debug output"
    )
    args = parser.parse_args()
    print(f"Starting with params: {args}")
    print(
        f"Starting with config: {dict(filter(lambda kv: not kv[0].startswith('__'),vars(config).items()))}"
    )

    return args


def prepare_dataloader(verbose):
    # Loading dataset
    print("loading dataset")
    print(config.dataset, config.dataset == "clotho")

    if config.dataset == "clotho":
        ds = Clotho(config.data_folder, subset="eval")
    elif config.dataset == "audiocaps":
        ds = AudioCaps(config.data_folder, subset="test")
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")

    if verbose:
        for i in range(len(ds)):
            try:
                print(i, ds[i]["fname"])
            except Exception as e:
                print(f"Exception on index {i}: {str(e)}")

    collate = BasicCollate()
    loader = DataLoader(ds, batch_size=1, collate_fn=collate)
    return loader


def inference(model: torch.nn.Module, data_loader):
    model.eval() if config.baseline_model == "conette" else model.clapcap.eval()
    predictions, references = [], []

    print("predicting eval dataset")

    start = perf_counter()
    if config.baseline_model == "clapcap":
        csv_path = (
            config.dataset == "clotho"
            and f"{config.data_folder}/CLOTHO_v2.1/clotho_csv_files/clotho_captions_evaluation.csv"
            or f"{config.data_folder}/AUDIOCAPS/csv_files_v1/test.csv"
        )
        audio_path = (
            config.dataset == "clotho"
            and f"{config.data_folder}/CLOTHO_v2.1/clotho_audio_files/evaluation"
            or f"{config.data_folder}/AUDIOCAPS/audio_32000Hz/test"
        )
        samples = build_samples(csv_path)
        start = perf_counter()
        filename, captions = next(iter(samples.items()))
        path = f"{audio_path}/{filename}"

        if os.path.exists(path):
            pred = model.generate_caption([path])
            predictions.extend(pred)
            references.append(captions)

    else:
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                audio = batch["audio"]
                sr = batch["sr"]
                # Process audio through model
                outputs = (
                    model(audio, sr, task="clotho")
                    if config.baseline_model == "conette"
                    else model(audio)
                )
                if i < 1:
                    print("model output:", outputs)
                candidates = outputs["cands"]

                # Collect predictions and references
                predictions.extend(candidates)
                references.extend(batch["captions"])
    end = perf_counter()
    inference_time = end - start
    return predictions, references, inference_time


def perform_inference(verbose, cpu):
    loader = prepare_dataloader(verbose)
    models_to_eval = prepare_models(loader)
    results = []

    for model in models_to_eval:
        torch_model = (
            model["model"]
            if config.baseline_model == "conette"
            else model["model"].clapcap
        )
        model_size_mb = get_model_size(torch_model)
        model_params = get_model_params(torch_model)
        print(
            f"starting inference on model {model['name']}, model size: {model_size_mb}"
        )

        predictions, references, inference_time = inference(
            model["model"], data_loader=loader
        )
        device = str(next(torch_model.parameters()).device)
        metadata = {
            "model": config.baseline_model,
            "compression_technique": model["name"],
            "model_size_mb": model_size_mb,
            "parameters": model_params,
            "device": device,
            "inference_time_in_s": f"{inference_time:.3f}",
            "predictions": predictions,
            "references": references,
        }
        results.append(metadata)
    return results


def load_previous_results(path):
    if not os.path.isfile(path):
        raise ValueError(f"{path} is not a file.")
    with open(path, "r") as fp:
        contents = json.load(fp)

    return [contents]


def save_results(results, fpath):
    os.makedirs("results", exist_ok=True)
    with open(
        fpath,
        "w",
    ) as fp:
        json.dump(results, fp, indent=2)


def save_inference_results(inference_results):
    for r in inference_results:
        save_results(
            r,
            f"results/inference_results_{r['compression_technique']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
        )


def perform_evaluation(inference_results):
    eval_results = []
    for r in inference_results:
        print(f"running evaluation on {r['model']} {r['compression_technique']}")
        predictions = r.pop("predictions")
        references = r.pop("references")
        corpus_scores, _ = evaluate(
            candidates=predictions,
            mult_references=references,
            metrics=config.metrics,
        )
        results = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in corpus_scores.items()
        }

        r["evaluation_results"] = results
        eval_results.append(r)

    return eval_results


def save_eval_results(eval_results):
    for r in eval_results:
        save_results(
            r,
            f"results/eval_results_{r['device']}_{r['compression_technique']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
        )


def main():
    if not config.inference and not config.evaluation:
        raise ValueError("Doing neither inference nor evaluation")

    args = parse_args()

    if config.inference:
        inference_results = perform_inference(args.verbose, args.cpu)
    elif args.path:
        inference_results = load_previous_results(args.path)
    else:
        raise ValueError("Inference was set to False while path argument is missing")

    if args.verbose:
        print("Inference results:")
        print(inference_results)

    if config.inference and config.save_inference_results:
        save_inference_results(inference_results)

    if config.evaluation:
        eval_results = perform_evaluation(inference_results)
        save_eval_results(eval_results)


if __name__ == "__main__":
    main()
