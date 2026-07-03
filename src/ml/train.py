from pathlib import Path
from ultralytics import YOLO

MODELS = ["yolo26l.pt", "yolo26s.pt"]
DEVICE = "cuda"
RESULT_DIR = Path("training_results")
YOLO_YAML_PATH = Path("dataset/BoneFractureYolo8/data.yaml")
RESULT_DIR.mkdir(exist_ok=True)
for model_name in MODELS:
    # Load a model
    model = YOLO(model_name)

    # Train the model
    model.train(data=YOLO_YAML_PATH, epochs=100, device=DEVICE)
