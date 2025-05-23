from conette import CoNeTTEConfig, CoNeTTEModel
from aac_datasets import Clotho, AudioCaps
import os
import torchaudio
import config


def main():
    # Download model
    model = CoNeTTEModel.from_pretrained(
        "Labbeti/conette", config=CoNeTTEConfig.from_pretrained("Labbeti/conette")
    )
    model.save_pretrained(config.model_folder + "baseline/")
    # Download dataset
    os.makedirs(config.data_folder, exist_ok=True)
    if len(config.download_clotho) > 0:
        if "eval" in config.download_clotho:
            Clotho(config.data_folder, subset="eval", download=True)
        if "val" in config.download_clotho:
            Clotho(config.data_folder, subset="val", download=True)
        if "dev" in config.download_clotho:
            Clotho(config.data_folder, subset="dev", download=True)
    if config.download_audiocaps:
        AudioCaps(config.data_folder, subset="val", download=True, verify_files=True)
        remove_corrupted_files()


def remove_corrupted_files():
    for filename in os.listdir(config.data_folder + "AUDIOCAPS/audio_32000Hz/val"):
        if filename.endswith(".flac"):
            file_path = os.path.join(
                config.data_folder + "AUDIOCAPS/audio_32000Hz/val", filename
            )
            try:
                # Attempt to load the audio file
                torchaudio.load(file_path)
            except Exception as e:
                # If an error occurs, delete the corrupted file
                print(f"Deleting corrupted file: {file_path}, Error: {e}")
                os.remove(file_path)


if __name__ == "__main__":
    main()
