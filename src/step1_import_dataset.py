import os

def download_data():
    LOCAL_PATH = "data/raw"
    os.makedirs(LOCAL_PATH, exist_ok=True)
    print("Downloading dataset...")
    # Note: Requires kaggle CLI configured
    os.system("kaggle datasets download mohsin31202/top-rated-movies-dataset -p data/raw --unzip")
    print("Dataset downloaded to:", LOCAL_PATH)

if __name__ == "__main__":
    download_data()