from pathlib import Path
from ultralytics import YOLO

MODELS = ["yolo26s.pt", "yolo26l.pt"]
DEVICE = "cuda"
YOLO_YAML_PATH = Path("dataset_grayscale/BoneFractureYolo8/data.yaml")
for model_name in MODELS:
    model = YOLO(model_name, verbose=True)
    model.train(
        data=YOLO_YAML_PATH,
        epochs=100,
        device=DEVICE,
        flipud=0.5,
        dropout=0.3,
        translate=0.5,
        shear=2,
        degrees=180,
        name=model_name.rstrip(".pt") + "_grayscale_high_res",
        exist_ok=True,
        workers=32,
        imgsz=1024,
    )
