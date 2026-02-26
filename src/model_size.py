import torch
import torch.nn as nn
import os


def get_model_size(model: torch.nn.Module, location: str = "DISK") -> float:
    """
    Measure model size in MB.
    CAUTION: DO NOT USE location="RAM" FOR QUANTIZED MODELS
    model.parameters() only include learnable parameters. Quantized parameters are not learnable, hence not returned, leading to an underestimation of the model size in RAM.
    Measuring model size on disk is more accurate, but it may not reflect the actual memory usage of the model during inference.
    Also, quantized parameters are not counted by model.parameters() and model.buffers(), so measuring their size in RAM will not reflect their actual memory usage.

    :param model: the model to measure
    :type model: torch.nn.Module
    :param location: whether to measure the size of the model in RAM or on disk
    :type location: str
    """
    if location == "RAM":
        # size of all learnable parameters
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        # size of all buffers (non-learnable parameters like running mean or variance in batch norm layers)
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size  # in bytes
        return total_size / (1024**2)  # Convert to MB
    elif location == "DISK":
        torch.save(model.state_dict(), "temp.p")
        size_mb_full = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return size_mb_full
    else:
        raise ValueError("location must be either 'RAM' or 'DISK'")


def get_model_params(model: torch.nn.Module):
    """
    Count the number of parameters in the model.
    CAUTION: DO NOT USE ON QUANTIZED MODELS

    Quantized parameters are not counted by model.parameters(), so this will not reflect the actual number of parameters in the model.
    Quantization doesn't change the amount of parameters anyway.

    :param model: the model to measure
    :type model: torch.nn.Module
    """
    return sum(p.numel() for p in model.parameters())


def count_qlinear_weight_bias_elems(qmodel: nn.Module):
    qlinear_type = torch.ao.nn.quantized.dynamic.Linear
    w_elems = 0
    b_elems = 0
    for _, mod in qmodel.named_modules():
        if isinstance(mod, qlinear_type):
            w, b = mod._packed_params._weight_bias()
            w_elems += w.numel()
            if b is not None:
                b_elems += b.numel()
    return w_elems, b_elems
