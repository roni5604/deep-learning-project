import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    """
    Downloads the Face Expression Recognition Dataset from Kaggle.
    Ensure that kaggle.json is placed in ~/.kaggle/.
    """
    api = KaggleApi()
    api.authenticate()
    dataset = "jonathanoheix/face-expression-recognition-dataset"
    output_dir = "data/face-expression-recognition-dataset/"
    os.makedirs(output_dir, exist_ok=True)
    api.dataset_download_files(dataset, path=output_dir, unzip=True)

if __name__ == "__main__":
    download_dataset()
