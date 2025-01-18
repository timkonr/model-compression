import torch
from conette import CoNeTTEConfig, CoNeTTEModel
from torch.utils.data import DataLoader
from aac_metrics import evaluate
from aac_datasets import Clotho
from aac_datasets.utils.collate import BasicCollate
import json
import os

print("loading dataset")
clotho_ev_ds = Clotho("data", subset="eval")
collate = BasicCollate()
loader = DataLoader(clotho_ev_ds, batch_size=32, collate_fn=collate)

# Step 1: Load Model
def load_model(model_path="Labbeti/conette", quantized=False):
    print("loading model")
    config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    if quantized:
        model = torch.quantization.quantize_dynamic(
            CoNeTTEModel.from_pretrained(model_path, config=config),
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    else:
        model = CoNeTTEModel.from_pretrained(model_path, config=config)
    return model

# Step 2: Prepare Dataset
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        # Load audio as a tensor
        audio_tensor = torch.load(audio_path)  # Replace with your audio loading logic
        return audio_tensor, label

# Step 3: Inference Function
def evaluate_model(model: CoNeTTEModel, data_loader):
    print("starting evaluation")
    model.eval()
    predictions, references = [], []
    
    print("loading previous predictions if available")
    if os.path.isfile("predictions") and os.path.getsize("predictions") > 0:
        with open("predictions", "r") as fp:
            predictions = json.load(fp)
            
    if os.path.isfile("references") and os.path.getsize("references") > 0:
        with open("references", "r") as fp:
            references = json.load(fp)

    if not len(predictions) > 0:
        print("predicting eval dataset")
        with torch.no_grad():
            for batch in data_loader:
                # Process audio through model
                outputs = model(batch["audio"], batch["sr"], task="clotho")
                candidates = outputs["cands"]

                # Collect predictions and references
                predictions.extend(candidates)
                references.extend(batch["captions"])
        print("saving predictions")
        with open("predictions", "w") as fp:
            json.dump(predictions, fp, indent=2)
        with open("references", "w") as fp:
            json.dump(references, fp, indent=2)
        
    # Evaluate using the metric
    print("running evaluation")
    corpus_scores, _ = evaluate(candidates=predictions, mult_references=references)
    return corpus_scores

# Step 4: Run Evaluation
if __name__ == "__main__":
    model = load_model()  # Load pre-trained model
    results = evaluate_model(model, data_loader=loader)
    print(f"Evaluation Results: {results}")
