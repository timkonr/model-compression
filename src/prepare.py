from conette import CoNeTTEConfig, CoNeTTEModel
from aac_datasets import Clotho, AudioCaps
import os
import utils.config as config
from msclap import CLAP


def main():
    if config.download_baseline_model:
        print("Downloading baseline models...")
        print("Downloading CoNeTTE...")
        model = CoNeTTEModel.from_pretrained(
            "Labbeti/conette", config=CoNeTTEConfig.from_pretrained("Labbeti/conette")
        )
        model.save_pretrained(config.model_folder + "baseline/")
        print("Downloading CLAP...")
        CLAP(version="clapcap")

    print("Downloading datasets...")
    os.makedirs(config.data_folder, exist_ok=True)
    if len(config.download_clotho) > 0:
        print("Downloading Clotho...")
        if "eval" in config.download_clotho:
            Clotho(config.data_folder, subset="eval", download=True)
        if "val" in config.download_clotho:
            Clotho(config.data_folder, subset="val", download=True)
        if "dev" in config.download_clotho:
            Clotho(config.data_folder, subset="dev", download=True)
    if len(config.download_audiocaps) > 0:
        print("Downloading AudioCaps...")
        [
            AudioCaps(
                config.data_folder,
                subset=subset,
                download=True,
                verify_files=True,
                max_workers=None,
                ytdlp_opts=[
                    "--cookies-from-browser",
                    f"{config.browser}{':' if len(config.browser_cookie_path) > 0 else ''}{config.browser_cookie_path}",
                ],
                audio_format="wav",
                sr=22050,
            )
            for subset in config.download_audiocaps
        ]


if __name__ == "__main__":
    main()
