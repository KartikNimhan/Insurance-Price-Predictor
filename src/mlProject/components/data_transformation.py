import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.mlProject.logger import logging
from src.mlProject.exception import CustomException
import sys

# Paths
raw_data_path = os.path.join("D:/My projects/Medical Insurane Predictor/data/raw_data", "insurance.csv")
processed_data_path = os.path.join("D:/My projects/Medical Insurane Predictor/data/Processed_data", "processed_data.csv")

# Ensure the directory for processed data exists
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

def preprocess_data():
    try:
        # Load the dataset
        df = pd.read_csv(raw_data_path)

        # Log the data shape
        logging.info(f"Loaded dataset with shape: {df.shape}")

        # Check for any missing values
        if df.isnull().sum().any():
            logging.warning(f"Missing values found in the dataset. Filling with mean or mode.")
            df.fillna(df.mean(), inplace=True)  # For numerical columns
            df.fillna(df.mode().iloc[0], inplace=True)  # For categorical columns

        # Encoding categorical columns using LabelEncoder
        label_encoder = LabelEncoder()

        df['sex'] = label_encoder.fit_transform(df['sex'])
        df['smoker'] = label_encoder.fit_transform(df['smoker'])
        df['region'] = label_encoder.fit_transform(df['region'])

        # Log the first few rows of the processed data
        logging.info(f"Processed data:\n{df.head()}")

        # Save the cleaned data to the processed folder
        df.to_csv(processed_data_path, index=False)
        logging.info(f"Data preprocessed and saved to {processed_data_path}")

    except Exception as e:
        logging.error(f"An error occurred during data preprocessing: {str(e)}")
        raise CustomException(str(e), sys)

if __name__ == "__main__":
    preprocess_data()
