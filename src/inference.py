from torch import qint8, tensor
from torch.nn import Linear
from torch.quantization import quantize_dynamic
from conette import CoNeTTEConfig, CoNeTTEModel

config = CoNeTTEConfig.from_pretrained("./model/")
model = CoNeTTEModel.from_pretrained("./model/", config=config)

model.to("cpu")
quantized_model = quantize_dynamic(model, {Linear}, dtype=qint8)
quantized_model.to("cpu")
path = "data/CLOTHO_v2.1/clotho_audio_files/evaluation/"

audio = [path + "105bpm.wav", path + "2013622thunder.wav"]
sr = [44100, 44100]

print(f"Model is on device: {next(quantized_model.parameters()).device}")
tensor = tensor([1.0])
print(f"Default tensor device: {tensor.device}")


outputs = quantized_model(audio, sr=sr, task="clotho")

print(outputs)
