from pathlib import Path
import os
from PIL import Image

from ultralytics import YOLO
from ultralytics.engine.results import Results


WEIGHTS_PATH = "runs/detect/train-5/weights/best.pt"
model = YOLO(WEIGHTS_PATH)

PATH = Path("dataset/BoneFractureYolo8/train/images")
paths = [str(PATH / f) for f in os.listdir(PATH)[:5]]
results: list[Results] = model(paths)

# Visualize the results
for i, r in enumerate(results):
    im_bgr = r.plot()
    im_rgb = Image.fromarray(im_bgr[..., ::-1])

    r.show()

    r.save(filename=f"results{i}.jpg")
