import torch
import torch.nn as nn
from pruner import ModelPruner
from train import train
from aac_datasets import Clotho
from torch.utils.data import DataLoader
from aac_datasets.utils.collate import BasicCollate
import torch_pruning as tp
from typing import Any, Iterable, Optional, Union
from torch import Tensor, Size
import config


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        # Inputs
        x: Union[Tensor, str, Iterable[str], Iterable[Tensor]],
        sr: Union[None, int, Iterable[int]] = None,
        x_shapes: Union[Tensor, None, list[Size]] = None,
        # Beam search options
        task: Union[str, list[str], None] = None,
        beam_size: Optional[int] = None,
        min_pred_size: Optional[int] = None,
        max_pred_size: Optional[int] = None,
        forbid_rep_mode: Optional[str] = None,
    ) -> dict[str, Any]:
        output = self.model(
            x=x,
            sr=sr,
            x_shapes=x_shapes,
            task=task,
            beam_size=beam_size,
            min_pred_size=min_pred_size,
            max_pred_size=max_pred_size,
            forbid_rep_mode=forbid_rep_mode,
        )
        return torch.as_tensor(output)  # , output

    def __call__(
        self,
        # Inputs
        x: Union[Tensor, str, Iterable[str], Iterable[Tensor]],
        sr: Union[None, int, Iterable[int]] = None,
        x_shapes: Union[Tensor, None, list[Size]] = None,
        # Beam search options
        task: Union[str, list[str], None] = None,
        beam_size: Optional[int] = None,
        min_pred_size: Optional[int] = None,
        max_pred_size: Optional[int] = None,
        forbid_rep_mode: Optional[str] = None,
    ) -> dict[str, Any]:
        return super().__call__(
            x=x,
            sr=sr,
            x_shapes=x_shapes,
            task=task,
            beam_size=beam_size,
            min_pred_size=min_pred_size,
            max_pred_size=max_pred_size,
            forbid_rep_mode=forbid_rep_mode,
        )


def use_torch_pruning(model: torch.nn.Module):
    loader = DataLoader(
        Clotho(config.data_folder, subset="eval"),
        batch_size=1,
        collate_fn=BasicCollate(),
    )
    for batch in loader:
        audio = batch["audio"]
        sr = batch["sr"]
        break

    imp = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 19788:
            ignored_layers.append(m)  # DO NOT prune the final classifier!

    iterative_steps = 5  # progressive pruning
    # model.config.use_cache = False
    if isinstance(audio, list):  # Convert list to tensor
        audio = torch.stack(audio)
    print(f"audio type: {type(audio)}, shape: {audio.shape}")
    output = model(audio)
    print(f"Output type: {type(output)}, output: {output}")

    wrapped_model = WrappedModel(model)

    pruner = tp.pruner.MagnitudePruner(
        wrapped_model,
        example_inputs=audio,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=0.2,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
        round_to=8,
    )
    base_macs, base_nparams = tp.utils.count_ops_and_params(wrapped_model, audio)
    # tp.utils.print_tool.before_pruning(wrapped_model)  # or print(model)
    wrapped_model.eval()
    for i in range(iterative_steps):
        pruner.step()
        # print(wrapped_model)
        # tp.utils.print_tool.after_pruning(wrapped_model)  # or print(model)
        macs, nparams = tp.utils.count_ops_and_params(wrapped_model, audio)
        print(f"MACs reduced from {base_macs} to {macs}")
        print(f"Params reduced from {base_nparams} to {nparams}")
    return wrapped_model  # , pruner, macs, nparams, audio, output

    # for i in range(iterative_steps):
    #     if isinstance(imp, tp.importance.TaylorImportance):
    #         # Taylor expansion requires gradients for importance estimation
    #         loss = model(audio)["preds"].sum()  # a dummy loss for TaylorImportance
    #         loss.backward()  # before pruner.step()
    #     pruner.step()
    #     macs, nparams = tp.utils.count_ops_and_params(model, audio)

    # fine tuning


def apply_structured_pruning(model: torch.nn.Module):
    # For structured pruning (removing entire channels/neurons)
    # Let's say we want to prune the least important 2 channels in conv layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            tp.ln_structured(
                module, name="weight", amount=0.2, n=2, dim=0
            )  # Prune 2 output channels
    return model


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
    model: torch.nn.Module, pruning_percentage=0.2, prune_biases=False
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

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            prune.remove(module, "weight")
            if prune_biases:
                prune.remove(module, "bias")

    return model


def prune_old(model, fine_tune=True):
    pruner = ModelPruner(model, prune_ratio=0.2)
    model = pruner.prune_model()
    if fine_tune:
        print("fine tuning pruned model")
        train_ds = Clotho(config.data_folder, subset="dev", download=True)
        val_ds = Clotho(config.data_folder, subset="val", download=True)

        collate = BasicCollate()
        train_loader = DataLoader(train_ds, batch_size=32, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate)
        model = train(model, train_loader, val_loader)
    return model


@torch.no_grad()
def prune(model: nn.Module, keep_ratio: float = 0.5, verbose: bool = True):
    """
    Structured pruning: shrink FFN hidden dim in every decoder layer by keeping top-k neurons.
    keep_ratio=0.5 means 2048 -> 1024.
    """
    model.eval()

    dec = model.model.decoder
    layers = dec.layers

    for li, layer in enumerate(layers):
        fc1: nn.Linear = layer.linear1
        fc2: nn.Linear = layer.linear2

        d_ff = fc1.out_features
        d_model = fc1.in_features

        if fc2.in_features != d_ff or fc2.out_features != d_model:
            raise RuntimeError(
                f"Unexpected shapes at layer {li}: "
                f"linear1 {tuple(fc1.weight.shape)}, linear2 {tuple(fc2.weight.shape)}"
            )

        k = int(round(d_ff * keep_ratio))
        k = max(1, min(k, d_ff))

        # Importance score per FFN neuron (row of fc1)
        scores = fc1.weight.norm(p=2, dim=1)
        keep_idx = torch.topk(scores, k=k, largest=True).indices
        keep_idx, _ = torch.sort(keep_idx)

        # Create new smaller linears
        new_fc1 = nn.Linear(d_model, k, bias=(fc1.bias is not None))
        new_fc2 = nn.Linear(k, d_model, bias=(fc2.bias is not None))

        # Copy weights/bias
        new_fc1.weight.copy_(fc1.weight[keep_idx, :])
        if fc1.bias is not None:
            new_fc1.bias.copy_(fc1.bias[keep_idx])

        # fc2.weight shape: (d_model, d_ff) -> keep columns
        new_fc2.weight.copy_(fc2.weight[:, keep_idx])
        if fc2.bias is not None:
            new_fc2.bias.copy_(fc2.bias)

        # Replace modules in-place
        layer.linear1 = new_fc1
        layer.linear2 = new_fc2

    return model
