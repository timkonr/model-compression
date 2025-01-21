import torch
import torch.nn.utils.prune as prune


def apply_unstructured_pruning(
    model: torch.nn.Module, pruning_percentage=0.3, prune_biases=False
):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=pruning_percentage)
            if prune_biases:
                prune.l1_unstructured(module, name="bias", amount=pruning_percentage)
    return model


def apply_global_unstructured_pruning(
    model: torch.nn.Module, pruning_percentage=0.3, prune_biases=False
):
    parameters_to_prune = []

    # Collect all layers to be pruned
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, "weight"))
            if prune_biases:
                parameters_to_prune.append((module, "bias"))

    # Apply global pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,  # Magnitude-based pruning
        amount=pruning_percentage,  # Proportion of weights to prune
    )

    return model
