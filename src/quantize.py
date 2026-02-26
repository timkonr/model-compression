import torch
import torch.nn as nn
import torch.ao.quantization as tq
from model_size import (
    count_qlinear_weight_bias_elems,
    get_model_params,
)


def make_quantized_model(
    model: torch.nn.Module, quantization_mode="dynamic", loader=None, dtype=torch.qint8
) -> torch.nn.Module:
    # GPU quantization doesn't work atm, because it is still in alpha or beta for pytorch
    # i.e. it works only for a fixed input shape,
    # which would require us to define a specific length for the input audio files and adapt the preprocessor in the process
    if quantization_mode == "dynamic":
        m = model.eval().to("cpu")
        total_params = get_model_params(m)
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=dtype)
        [w, b] = count_qlinear_weight_bias_elems(m)
        print(
            f"Quantized layers have {w} quantized weight elements and {b} bias elements and {total_params - w - b} non-quantized parameters"
        )
        return m
    else:
        return make_static_quantized_model(model, loader=loader)


def make_static_quantized_model(model: torch.nn.Module, loader=None):
    wrapped = StaticQuantWrapper(model).eval().cpu()
    # model.eval()
    wrapped.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    for name, m in wrapped.named_modules():
        if (
            name.endswith("self_attn")
            or "spectrogram_extractor" in name.lower()
            or "stft" in name.lower()
            or isinstance(
                m,
                (
                    torch.nn.Embedding,
                    torch.nn.EmbeddingBag,
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                ),
            )
        ):
            m.qconfig = None
        else:
            m.qconfig = torch.ao.quantization.get_default_qconfig("x86")
    # model = torch.ao.quantization.fuse_modules(model, [["conv", "relu"]])
    wrapped = torch.ao.quantization.prepare(wrapped)
    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    calibrate_model(wrapped, data_loader=loader)
    wrapped = torch.ao.quantization.convert(wrapped)
    return wrapped


def calibrate_model(model: torch.nn.Module, data_loader, num_batches=50):
    model.eval()
    model.cpu()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i < 1:
                find_list_observer_inputs(model, batch["audio"], batch["sr"])
            if i >= num_batches:
                break

            audio = batch["audio"]
            sr = batch["sr"]

            model(audio, sr, task="clotho")


class StaticQuantWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, audio, sr, task="clotho"):
        audio = self.quant(audio)
        out = self.base(audio, sr, task=task)

        if isinstance(out, dict):
            y = next(v for v in out.values() if torch.is_tensor(v))
        else:
            y = out
        return self.dequant(y)


def find_list_observer_inputs(prepared_model, audio, sr):
    bad = []

    def hook(mod, inp, out):
        x = inp[0] if isinstance(inp, tuple) and len(inp) else inp
        if isinstance(x, list):
            bad.append((mod.__class__.__name__, type(x), len(x)))

    handles = []
    for name, m in prepared_model.named_modules():
        if "observer" in m.__class__.__name__.lower():
            handles.append(m.register_forward_hook(hook))

    with torch.no_grad():
        prepared_model(audio, sr)

    for h in handles:
        h.remove()

    return bad
