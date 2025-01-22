from torch import qint8, tensor
from torch.nn import Linear
from torch.quantization import quantize_dynamic
from conette import CoNeTTEConfig, CoNeTTEModel
from utils import load_model, get_model_size

path = "data/CLOTHO_v2.1/clotho_audio_files/evaluation/"

audio = [path + "105bpm.wav", path + "2013622thunder.wav"]
sr = [44100, 44100]

models = [
    {"model": load_model(), "name": "baseline"},
    {"model": load_model(quantized=True), "name": "quantized"},
    {"model": load_model(pruned=True), "name": "pruned"},
]

outputs = [
    {
        "output": model["model"](audio, sr=sr, task="clotho")["cands"],
        "model": model["name"],
        "model_size": get_model_size(model["model"]),
    }
    for model in models
]

print(outputs)
