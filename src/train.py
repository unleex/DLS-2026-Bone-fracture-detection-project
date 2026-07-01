from pathlib import Path
import json
from ultralytics import YOLO

MODELS = ["yolo26s.pt", "yolo26l.pt"]
DEVICE = "cuda"
RESULT_DIR = Path("training_results")
DATA_PATH = Path("dataset/VisDrone.yaml")
RESULT_DIR.mkdir(exist_ok=True)
for model_name in MODELS:
    # Load a model
    model = YOLO(model_name)

    # Train the model
    results = model.train(data=DATA_PATH, epochs=100, device=DEVICE)
    json.dump(results, open(str(RESULT_DIR / model_name) + ".json", "w+"), indent="\t")
