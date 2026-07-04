import shutil
import kagglehub
from PIL import Image
import os
from pathlib import Path

path = "dataset"
# # Download latest version
# path = kagglehub.dataset_download(
# "pkdarabi/bone-fracture-detection-computer-vision-project", output_dir=path
# )
print("Path to dataset files:", path)

print("Converting to grayscale...")
grayscale_path = Path(path + "_grayscale")
shutil.copytree(path, str(grayscale_path))

for file in grayscale_path.rglob("*"):
    try:
        img = Image.open(file).convert("L")  # Convert to grayscale
    except:
        continue
    img.save(file)
