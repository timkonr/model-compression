import json
from aac_metrics import evaluate

predictions = []
references = []

with open("logs/predictions_quantized", "r") as fp:
    predictions = json.load(fp)

with open("logs/references_quantized", "r") as fp:
    references = json.load(fp)

print(predictions)

print("running evaluation")
corpus_scores, _ = evaluate(candidates=predictions, mult_references=references, metrics="all")
results = {key: value.item() for key, value in corpus_scores.items()}
with open(f"results/eval_results_quantized", "w") as fp:
    json.dump(results, fp, indent=2)