import os
import zipfile
import kagglehub

LOCAL_PATH = "data/raw"
os.makedirs(LOCAL_PATH, exist_ok=True)

# Download dataset
os.system(
    "kaggle datasets download mohsin31202/top-rated-movies-dataset -p data/raw --unzip"
)

print("Dataset downloaded to:", LOCAL_PATH)
