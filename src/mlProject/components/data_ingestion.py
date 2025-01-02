import os
import requests
from src.mlProject.logger import logging
from src.mlProject.exception import CustomException
import sys

# Path for saving the downloaded CSV
raw_data_path = os.path.join("D:/My projects/Medical Insurane Predictor/data/raw_data", "insurance.csv")

def download_data():
    try:
        file_id = '1DhNsRHCLzyZTe_zGg8JwuokbKjdS4C7W'  # Replace with your actual file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(url)

        if response.status_code == 200:
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            with open(raw_data_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Data downloaded successfully to {raw_data_path}")
        else:
            logging.error(f"Failed to download data. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"An error occurred while downloading data: {str(e)}")
        raise CustomException(str(e), sys)

if __name__ == "__main__":
    download_data()
