from torch import qint8
from torch.nn import Linear
from torch.quantization import quantize_dynamic
from conette import CoNeTTEConfig, CoNeTTEModel

config = CoNeTTEConfig.from_pretrained("./model/")
model = CoNeTTEModel.from_pretrained("./model/", config=config)

quantized_model = quantize_dynamic(
    model, {Linear}, dtype=qint8
)
quantized_model.to("cpu")
path = "data/CLOTHO_v2.1/clotho_audio_files/evaluation/"
outputs = quantized_model([path + "105bpm.wav", path + "2013622thunder.wav"], sr=[44100, 44100], task="clotho")

print(outputs)
