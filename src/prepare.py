from conette import CoNeTTEConfig, CoNeTTEModel
from aac_datasets import Clotho
import os

def main():
    # Download model
    config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config)
    model.save_pretrained("./model/")
    
    # Download dataset
    os.makedirs("data", exist_ok=True)
    clotho_ev_ds = Clotho("data", subset="eval", download=True)

if __name__ == "__main__":
    main()
