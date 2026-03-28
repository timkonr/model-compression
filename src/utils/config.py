# setup config
download_baseline_model = True
download_audiocaps = ["test"]  # add train and val for training
download_clotho = ["eval"]  # add dev and val for training
browser = "firefox"  # for downloading audiocaps with yt-dlp, e.g. "chrome" or "edge"
browser_cookie_path = ""  # optional path to cookies. for more info see https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp

# evaluation config
baseline_model = "conette"  # clapcap | conette
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
quantization = False  # Inference on quantized model
quantization_mode = "dynamic"  # dynamic | static
pruning = True  # Inference on pruned model
kd = False  # Inference on saved kd model
kd_model = "best_student_model.pth"  # Path to kd model
data_folder = "data/"  # Path to data folder
model_folder = "model/"  # Path to model folder

# pruning config
## conette
convnext_3072_keep_ratio = 0.5
convnext_1536_keep_ratio = 0.875
decoder_keep_ratio = 0.5
pruning_score_mode = "sum_l2"  # sum_l2 (consider in and out strength) | first_l2 (consider only in strength)

## clapcap
gpt_keep_ratio = None  # nach 6 stunden abgebrochen
mapper_keep_ratio = 0.5
htsat_keep_ratio = 0.75
htsat_min_hidden_dim = 1536


# kd config
patience = 5
num_epochs = 25
batch_size = 32
