from pathlib import Path
from ultralytics import YOLO

MODELS = [
    "yolo26x.pt",
    "yolo26s.pt",
]
DEVICE = "cuda"
BATCH_SIZE = 64
YOLO_YAML_PATH = Path("dataset/VisDrone.yaml")
for model_name in MODELS:
    model = YOLO(model_name, verbose=True)
    model.train(
        data=YOLO_YAML_PATH,
        epochs=100,
        device=DEVICE,
        name=model_name.rstrip(".pt") + "_high_res",
        exist_ok=True,
        workers=32,
        cos_lr=True,
        imgsz=1024,
        batch=BATCH_SIZE,
        # resume=True,
    )
