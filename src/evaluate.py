import torch
from conette import CoNeTTEModel
from torch.utils.data import DataLoader
from aac_metrics import evaluate
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
import json
import argparse
from time import perf_counter
from utils import get_model_size, load_model


def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument(
        "--baseline", action="store_true", default=True, help="Evaluate baseline model."
    )
    parser.add_argument(
        "--no-baseline",
        dest="baseline",
        action="store_false",
        help="Do not evaluate baseline model.",
    )
    parser.add_argument(
        "--quantization", action="store_true", help="Evaluate quantized model."
    )
    parser.add_argument("--pruning", action="store_true", help="Evaluate pruned model.")
    args = parser.parse_args()
    print(f"Starting with params: {args}")

    # Loading dataset
    print("loading dataset")
    clotho_ev_ds = Clotho("data", subset="eval")
    collate = BasicCollate()
    loader = DataLoader(clotho_ev_ds, batch_size=32, collate_fn=collate)

    # Load models
    models_to_eval = []
    if args.baseline:
        models_to_eval.append({"model": load_model(), "name": "baseline"})
    if args.quantization:
        models_to_eval.append(
            {"model": load_model(quantized=True), "name": "quantized"}
        )
    if args.pruning:
        models_to_eval.append({"model": load_model(pruned=True), "name": "pruned"})

    # Evaluate models
    for model in models_to_eval:
        model_size_mb = get_model_size(model["model"])
        results, inference_time = evaluate_model(
            model["model"],
            data_loader=loader,
            quantized=True if "quantized" in model["name"] else False,
        )
        results = {key: value.item() for key, value in results.items()}
        metadata = {
            "model_name": model["name"],
            "model_size_mb": model_size_mb,
            "device": str(next(model["model"].parameters()).device),
            "inference_time_in_s": f"{inference_time:.3f}",
            "evaluation_results": results,
        }

        with open(f"results/eval_results_{model['name']}", "w") as fp:
            json.dump(metadata, fp, indent=2)
        print(f"Evaluation Results for {model['name']} model: {results}")


def evaluate_model(model: CoNeTTEModel, data_loader, quantized=False):
    print("starting evaluation")
    model.eval()
    predictions, references = [], []

    # Set device
    # device = (
    #     torch.device("cpu")
    #     if quantized
    #     else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # )
    # model.to(device)

    print("predicting eval dataset")

    start = perf_counter()
    with torch.no_grad():
        for batch in data_loader:
            # Move batch tensors to the CPU for quantized model
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
    # Evaluate using the metric
    print("running evaluation")
    corpus_scores, _ = evaluate(
        candidates=predictions, mult_references=references, metrics="all"
    )
    return corpus_scores, inference_time


if __name__ == "__main__":
    main()
