import torch
from utils.model_size import (
    count_qlinear_weight_bias_elems,
    get_model_params,
)
from utils import config
from torchao.quantization import Int4WeightOnlyConfig, quantize_


def make_quantized_model(model: torch.nn.Module, dtype=torch.qint8) -> torch.nn.Module:
    # GPU quantization doesn't work atm, because it is still in alpha or beta for pytorch
    # i.e. it works only for a fixed input shape,
    # which would require us to define a specific length for the input audio files and adapt the preprocessor in the process
    m = model if config.baseline_model == "conette" else model.clapcap
    m.eval()
    total_params = get_model_params(m)

    if config.baseline_model == "conette":
        if torch.cuda.is_available():
            quantize_(m, Int4WeightOnlyConfig())
            m = torch.compile(m, mode="max-autotune", fullgraph=True)
        else:
            m.cpu()
            m = torch.quantization.quantize_dynamic(
                m, {torch.nn.Linear}, dtype=dtype, inplace=True
            )
    elif config.baseline_model == "clapcap":
        # for clapcap we only quantize the encoder (m.clap) and the CLAP projection (m.clap_project)
        if torch.cuda.is_available():
            quantize_(model.clap, Int4WeightOnlyConfig())
            quantize_(model.clap_project, Int4WeightOnlyConfig())
        else:
            torch.quantization.quantize_dynamic(
                m.clap, {torch.nn.Linear}, dtype=dtype, inplace=True
            )
            torch.quantization.quantize_dynamic(
                m.clap_project, {torch.nn.Linear}, dtype=dtype, inplace=True
            )
    w, b, n_layers = count_qlinear_weight_bias_elems(m)
    print(
        f"Quantization summary: {n_layers} Linear layers quantized | "
        f"{w} quantized weight elements | "
        f"{b} bias elements | "
        f"{total_params - w - b} parameters in non-quantized layers"
    )
    return model
