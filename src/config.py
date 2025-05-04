dataset = "clotho"  # clotho | audiocaps
download_audiocaps = False  # used only in prepare script
download_clotho = [
    "eval",
]  # used only in prepare script. Use dev and val for training, eval for evaluation
metrics = "all"  # see aac_metrics.functional.evaluate
inference = True
save_inference_results = True
evaluation = True
baseline = False  # Inference on baseline model
quantization = False  # Inference on quantized model
pruning = False  # Inference on pruned model
kd = True  # Inference on saved kd model
kd_model = "best_student_model.pth"  # Path to kd model
data_folder = "data/"  # Path to data folder
model_folder = "model/"  # Path to model folder

patience = 5
num_epochs = 25
batch_size = 32
