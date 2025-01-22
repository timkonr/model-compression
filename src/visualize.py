import json

metrics_to_evaluate = ["fense", "spider", "meteor"]

baseline = {}
quantized = {}
pruned = {}

with open("results/eval_results_baseline", "r") as fp:
    baseline = json.load(fp)

with open("results/eval_results_quantized", "r") as fp:
    quantized = json.load(fp)

with open("results/eval_results_pruned", "r") as fp:
    pruned = json.load(fp)


baseline_results = {
    key: value
    for key, value in baseline["evaluation_results"].items()
    if key in metrics_to_evaluate
}
quantized_results = {
    key: value
    for key, value in quantized["evaluation_results"].items()
    if key in metrics_to_evaluate
}
pruned_results = {
    key: value
    for key, value in pruned["evaluation_results"].items()
    if key in metrics_to_evaluate
}


differences = {
    key: baseline_results[key] - quantized_results[key] for key in baseline_results
}
# max_diff = max(differences, key=differences.get)
# print(
#     f"Metric with highest deviation in quantized model: {max_diff}\nDifference: {differences[max_diff]}"
# )
print(f"Differences in metrics between baseline and quantized models:\n{differences}")

differences = {
    key: baseline_results[key] - pruned_results[key] for key in baseline_results
}
# max_diff = max(differences, key=differences.get)
# print(
#     f"Metric with highest deviation in pruned model: {max_diff}\nDifference: {differences[max_diff]}"
# )
print(f"Differences in metrics between baseline and pruned models:\n{differences}")
