# setup config
download_baseline_model = True
download_audiocaps = True
download_clotho = [
    # "eval",
]  # used only in prepare script. Use dev and val for training, eval for evaluation
browser = "firefox"  # for downloading audiocaps with yt-dlp, e.g. "chrome" or "edge"
browser_cookie_path = ""  # optional path to cookies. for more info see https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp

# evaluation config
baseline_model = "clapcap"  # clapcap | conette
dataset = "audiocaps"  # clotho | audiocaps
metrics = (
    "meteor",
    "spider",
    "fense",
    "vocab",
)
inference = True
save_inference_results = True
evaluation = True
baseline = False  # Inference on baseline model
quantization = True  # Inference on quantized model
quantization_mode = "dynamic"  # dynamic | static
pruning = False  # Inference on pruned model
kd = False  # Inference on saved kd model
kd_model = "best_student_model.pth"  # Path to kd model
data_folder = "data/"  # Path to data folder
model_folder = "model/"  # Path to model folder

# kd config
patience = 5
num_epochs = 25
batch_size = 32
