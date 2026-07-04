from pathlib import Path
from ultralytics import YOLO

DEVICE = "cuda"
YOLO_YAML_PATH = Path("dataset/BoneFractureYolo8/data.yaml")

model = YOLO("yolov8m.pt")

model.train(task="detect", mode="train", device=DEVICE, data=YOLO_YAML_PATH, epochs=60)
