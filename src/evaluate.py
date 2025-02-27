import torch
from conette import CoNeTTEModel
from torch.utils.data import DataLoader
from aac_metrics import evaluate
from aac_datasets import Clotho, AudioCaps
from aac_datasets.utils.collate import BasicCollate
import json
import argparse
from time import perf_counter
from utils import get_model_size, load_model, get_model_params
import config
import datetime
import os


def main():
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
    parser.add_argument("-v", "--verbose", help="Print some debug output")
    args = parser.parse_args()
    print(f"Starting with params: {args}")
    print(f"Starting with config: {config}")

    if config.inference:
        # Loading dataset
        print("loading dataset")
        print(config.dataset, config.dataset == "clotho")

        if config.dataset == "clotho":
            ds = Clotho("data", subset="eval")
        elif config.dataset == "audiocaps":
            ds = AudioCaps("data", subset="val")
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset}")

        if args.verbose:
            for i in range(len(ds)):
                try:
                    print(i, ds[i]["fname"])
                except Exception as e:
                    print(f"Exception on index {i}: {str(e)}")

        collate = BasicCollate()
        loader = DataLoader(ds, batch_size=1, collate_fn=collate)

        # Load models
        models_to_eval = []
        if config.baseline:
            models_to_eval.append({"model": load_model(), "name": "baseline"})
        if config.quantization:
            models_to_eval.append(
                {"model": load_model(quantized=True), "name": "quantized"}
            )
        if config.pruning:
            models_to_eval.append({"model": load_model(pruned=True), "name": "pruned"})

        # Evaluate models
        for model in models_to_eval:
            if args.cpu:
                model["model"].to("cpu")
            model_size_mb = get_model_size(model["model"])
            model_params = get_model_params(model["model"])
            results, inference_time = evaluate_model(model["model"], data_loader=loader)
            results = {key: value.item() for key, value in results.items()}
            device = str(next(model["model"].parameters()).device)
            metadata = {
                "model_name": model["name"],
                "model_size_mb": model_size_mb,
                "parameters": model_params,
                "dataset": config.dataset,
                "device": device,
                "inference_time_in_s": f"{inference_time:.3f}",
                "evaluation_results": results,
            }

            with open(
                f"results/eval_results_{device}_{model['name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
                "w",
            ) as fp:
                json.dump(metadata, fp, indent=2)
            print(f"Evaluation Results for {model['name']} model: {results}")
    elif config.evaluation:
        # do evaluation without inference
        print(f"Evaluating results from: {args.path}")
        evaluate_model(path=args.path)
    else:
        raise ValueError("Doing neither inference nor evaluation")


def inference(model: CoNeTTEModel, data_loader):
    print("starting inference")
    model.eval()
    predictions, references = [], []

    print("predicting eval dataset")
    start = perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            print(f"Batch {i}: {batch['fname']}")
            audio = batch["audio"]
            sr = batch["sr"]

            # Process audio through model
            outputs = model(audio, sr, task="clotho")
            candidates = outputs["cands"]

            # Collect predictions and references
            predictions.extend(candidates)
            references.extend(batch["captions"])

    end = perf_counter()
    inference_time = end - start
    return predictions, references, inference_time


def parse_previous_results(path: str):
    if not os.path.isfile(path):
        raise ValueError(f"{path} is not a file.")
    with open(f"path", "r") as fp:
        contents = json.load(fp)
        predictions = contents.predictions
        references = contents.references
        inference_time = contents.inference_time

    return predictions, references, inference_time


def evaluate_model(model: CoNeTTEModel, data_loader, path):
    predictions, references, inference_time = (
        inference(model, data_loader)
        if config.inference
        else parse_previous_results(path)
    )

    if config.evaluation:
        # Evaluate using the metric
        print("running evaluation")
        corpus_scores, _ = evaluate(
            candidates=predictions, mult_references=references, metrics=config.metrics
        )
        return corpus_scores, inference_time
    else:
        # Save inference results
        metadata = {
            "model_name": model["name"],
            "inference_time_in_s": f"{inference_time:.3f}",
            "predictions": predictions,
            "references": references,
        }
        with open(
            f"results/inference_results_{model['name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}",
            "w",
        ) as fp:
            json.dump(metadata, fp, indent=2)


if __name__ == "__main__":
    main()
