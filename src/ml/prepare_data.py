import kagglehub

path = "dataset"
# Download latest version
path = kagglehub.dataset_download("kushagrapandya/visdrone-dataset", output_dir=path)
print("Path to dataset files:", path)
