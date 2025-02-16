import torch
from conette import CoNeTTEConfig, CoNeTTEModel
from prune import apply_global_unstructured_pruning


def load_model(model_path="./model/", quantized=False, pruned=False):
    print("loading model")
    config = CoNeTTEConfig.from_pretrained(model_path)
    if quantized:
        model = CoNeTTEModel.from_pretrained(model_path, config=config)
        # model.to("cpu")
        # model = torch.quantization.quantize_dynamic(
        #     model, {torch.nn.Linear}, dtype=torch.qint8
        # )
        # model.to("cpu")
    else:
        model = CoNeTTEModel.from_pretrained(model_path, config=config)
        if pruned:
            apply_global_unstructured_pruning(model)
    return model


def get_model_size(model: torch.nn.Module):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size  # in bytes
    return total_size / (1024**2)  # Convert to MB


def get_model_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def check_sparsity(model: torch.nn.Module):
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    print(f"Total parameters: {total_params}")
    print(f"Zero parameters: {zero_params}")
    print(f"Sparsity: {zero_params / total_params:.2%}")
