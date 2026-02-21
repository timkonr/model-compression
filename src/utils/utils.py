from conette import CoNeTTEConfig, CoNeTTEModel
from prune import prune
from utils import config
from quantize import make_quantized_model
from student_model import load_student_model
from model_size import get_model_size, get_model_params
from torch.utils.data import DataLoader


def prepare_models(loader: DataLoader):
    # Load models
    models_to_eval = []
    if config.baseline:
        models_to_eval.append({"model": load_model(), "name": "baseline"})
    if config.quantization:
        models_to_eval.append(
            {"model": load_model(quantized=True, loader=loader), "name": "quantized"}
        )
    if config.pruning:
        models_to_eval.append({"model": load_model(pruned=True), "name": "pruned"})
    if config.kd:
        models_to_eval.append({"model": load_model(kd=True), "name": "kd"})
    return models_to_eval


def load_model(
    model_path=config.model_folder,
    quantized=False,
    pruned=False,
    kd=False,
    loader=None,
    verbose=True,
):
    print("loading model")
    baseline_path = model_path + "baseline/"
    model = CoNeTTEModel.from_pretrained(
        baseline_path, config=CoNeTTEConfig.from_pretrained(baseline_path)
    )
    model.to("cpu")
    if verbose:
        print(f"Original model size on disk: {get_model_size(model):.2f} MB")
        print(f"Original model params: {get_model_params(model)}")
    if kd:
        model = load_student_model()
    if quantized:
        model = make_quantized_model(
            model, quantization_mode=config.quantization_mode, loader=loader
        )
        for name, m in model.named_modules():
            cls = m.__class__
            if cls.__name__ in {
                "Linear",
                "NonDynamicallyQuantizableLinear",
            }:
                print(name, "->", cls.__module__ + "." + cls.__name__)
    if pruned:
        model = prune(model, keep_ratio=0.5)
    model.to("cpu")  # ensure model is on CPU
    if verbose:
        new_model_type = (
            "Pruned and quantized"
            if quantized and pruned
            else "Pruned" if pruned else "Quantized" if quantized else "original"
        )
        print(f"{new_model_type} model size on disk: {get_model_size(model):.2f} MB")
        print(f"{new_model_type} model params: {get_model_params(model)}")
    return model
