import torch
from conette import CoNeTTEConfig, CoNeTTEModel
from torch.utils.data import DataLoader
from aac_metrics import evaluate
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
import json
import argparse

def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("--baseline", type=bool, default=True, help="Evaluate baseline model.")
    parser.add_argument("--quantization", type=bool, default=False, help="Evaluate quantized model.")
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
        models_to_eval.append({"model": load_model(), "name":"baseline"})
    if args.quantization:
        models_to_eval.append({"model": load_model(quantized=True), "name": "quantized"})
    
    # Evaluate models
    for model in models_to_eval:
        model_size_mb = get_model_size(model["model"])
        results = evaluate_model(model["model"], data_loader=loader, quantized=True if "quantized" in model["name"] else False)
        results = {key: value.item() for key, value in results.items()}
        metadata = {
        "model_name": model["name"],
        "model_size_mb": model_size_mb,
        "evaluation_results": results
        }

        with open(f"results/eval_results_{model['name']}", "w") as fp:
            json.dump(metadata, fp, indent=2)
        print(f"Evaluation Results for {model['name']} model: {results}")

def get_model_size(model: torch.nn.Module):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size  # in bytes
    return total_size / (1024 ** 2)  # Convert to MB

def load_model(model_path="Labbeti/conette", quantized=False):
    print("loading model")
    config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    if quantized:
        model = torch.quantization.quantize_dynamic(
            CoNeTTEModel.from_pretrained(model_path, config=config),
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        model.to("cpu")
    else:
        model = CoNeTTEModel.from_pretrained(model_path, config=config)
    return model

def evaluate_model(model: CoNeTTEModel, data_loader, quantized=False):
    print("starting evaluation")
    model.eval()
    predictions, references = [], []
    
    print("predicting eval dataset")
    with torch.no_grad():
        for batch in data_loader:
            # Move batch tensors to the CPU for quantized model
            audio = batch["audio"] if isinstance(batch["audio"], list) else batch["audio"].tolist()
            sr = batch["sr"] if isinstance(batch["sr"], list) else batch["sr"].tolist()
            
            # Process audio through model
            outputs = model(audio, sr, task="clotho")
            candidates = outputs["cands"]

            # Collect predictions and references
            predictions.extend(candidates)
            references.extend(batch["captions"])
        
    # Evaluate using the metric
    print("running evaluation")
    corpus_scores, _ = evaluate(candidates=predictions, mult_references=references, metrics="all")
    return corpus_scores

if __name__ == "__main__":
    main()
