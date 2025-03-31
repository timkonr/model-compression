import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelPruner:
    def __init__(self, model, prune_ratio=0.2):
        self.model = model
        self.prune_ratio = prune_ratio

    def compute_l1_norm(self, conv_layer):
        """Compute L1 norm of each filter in a Conv2d layer."""
        return torch.abs(conv_layer.weight).sum(dim=(1, 2, 3))

    def prune_conv_layer(self, conv_layer):
        """Prune filters from a Conv2d layer based on L1 norm."""
        num_filters = conv_layer.out_channels
        num_prune = int(self.prune_ratio * num_filters)

        if num_prune == 0:
            return conv_layer, torch.arange(num_filters)

        l1_norms = self.compute_l1_norm(conv_layer)
        prune_indices = torch.argsort(l1_norms)[
            :num_prune
        ]  # Prune lowest L1 norm filters
        keep_indices = torch.argsort(l1_norms)[
            num_prune:
        ]  # Keep highest L1 norm filters

        new_weight = torch.index_select(conv_layer.weight, 0, keep_indices)

        new_conv = nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=len(keep_indices),
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            bias=(conv_layer.bias is not None),
        )

        new_conv.weight.data = new_weight.clone()
        if conv_layer.bias is not None:
            new_conv.bias.data = torch.index_select(conv_layer.bias, 0, keep_indices)

        return new_conv, keep_indices

    def prune_batchnorm_layer(self, bn_layer, keep_indices):
        """Prune a BatchNorm2d layer to match the new number of channels."""
        new_bn = nn.BatchNorm2d(len(keep_indices))

        new_bn.weight.data = torch.index_select(
            bn_layer.weight, 0, keep_indices
        ).clone()
        new_bn.bias.data = torch.index_select(bn_layer.bias, 0, keep_indices).clone()
        new_bn.running_mean.data = torch.index_select(
            bn_layer.running_mean, 0, keep_indices
        ).clone()
        new_bn.running_var.data = torch.index_select(
            bn_layer.running_var, 0, keep_indices
        ).clone()

        return new_bn

    def prune_model(self, module=None):
        """Apply pruning recursively to all Conv2d and BatchNorm2d layers in the model."""
        if module is None:
            module = self.model

        prev_keep_indices = (
            None  # Store keep indices for corresponding batch norm layers
        )

        for name, submodule in module.named_children():
            if isinstance(submodule, nn.Conv2d):
                new_conv, keep_indices = self.prune_conv_layer(submodule)
                setattr(module, name, new_conv)
                prev_keep_indices = (
                    keep_indices  # Store indices for corresponding BatchNorm layer
                )
            elif (
                isinstance(submodule, nn.BatchNorm2d) and prev_keep_indices is not None
            ):
                new_bn = self.prune_batchnorm_layer(submodule, prev_keep_indices)
                setattr(module, name, new_bn)
                prev_keep_indices = None  # Reset after use
            else:
                self.prune_model(submodule)  # Recurse into submodules

        return self.model


# Example usage:
# model = torch.load("trained_model.pth")
# pruner = ModelPruner(model, prune_ratio=0.2)
# pruned_model = pruner.prune_model()
# torch.save(pruned_model, "pruned_model.pth")
