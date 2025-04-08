import torch
from conette import CoNeTTEConfig, CoNeTTEModel
from prune import use_torch_pruning
import config


def load_model(model_path="./model/", quantized=False, pruned=False, kd=False):
    print("loading model")
    model = CoNeTTEModel.from_pretrained(
        model_path, config=CoNeTTEConfig.from_pretrained(model_path)
    )
    if kd:
        model = torch.load(f"{model_path}{config.kd_model}")
    if quantized:
        # GPU quantization doesn't work atm, because it is still in alpha or beta for pytorch
        # i.e. it actually would work, but only for a fixed input shape,
        # which would require us to define a specific length for the input audio files and adapt the preprocesser in the process
        # so I think this is beyond the scope of this thesis
        model.to("cpu")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        model.to("cpu")
    if pruned:
        print(f"Initial model size: {get_model_size(model)}")
        print(f"Initial model params: {get_model_params(model)}")
        # model = prune(model, fine_tune=False)

        use_torch_pruning(model)
        print(f"Pruned model size: {get_model_size(model)}")
        print(f"Pruned model params: {get_model_params(model)}")
        # apply_global_unstructured_pruning(model)
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
