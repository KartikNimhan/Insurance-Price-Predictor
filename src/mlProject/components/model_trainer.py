# import os
# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import LabelEncoder
# import logging
# import sys

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Paths
# processed_data_path = "D:/My projects/Medical Insurane Predictor/data/Processed_data/processed_data.csv"
# model_save_path = "D:/My projects/Medical Insurane Predictor/notebook/mlruns/models/insurance_price_predictor.pkl"

# def train_model():
#     try:
#         mlflow.set_tracking_uri("http://localhost:5000")

#         # Load the preprocessed data
#         df = pd.read_csv(processed_data_path)
#         logging.info("Data loaded successfully.")

#         # Handle missing values
#         df = df.ffill()
#         logging.info("Missing values handled.")

#         # Encode categorical variables
#         for col in ['sex', 'smoker', 'region']:
#             if col in df.columns:
#                 encoder = LabelEncoder()
#                 df[col] = encoder.fit_transform(df[col])
#         logging.info("Categorical variables encoded.")

#         # Split features and target variable
#         X = df.drop(columns=['charges'])
#         y = df['charges']

#         # Train-test split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Initialize the model
#         model = LinearRegression()

#         # Start MLflow experiment
#         with mlflow.start_run():
#             # Train the model
#             model.fit(X_train, y_train)
#             logging.info("Model training completed.")

#             # Make predictions and calculate metrics
#             y_pred = model.predict(X_test)
#             mse = mean_squared_error(y_test, y_pred)
#             logging.info(f"Mean Squared Error: {mse}")

#             # Log parameters, metrics, and the model with MLflow
#             mlflow.log_param("model_type", "Linear Regression")
#             mlflow.log_metric("mean_squared_error", mse)
#             mlflow.sklearn.log_model(model, "insurance_price_predictor")
#             logging.info("Model logged with MLflow.")

#         # Save the model as a pickle file
#         import joblib
#         joblib.dump(model, model_save_path)
#         logging.info(f"Model saved at {model_save_path}")

#     except Exception as e:
#         logging.error(f"An error occurred during model training: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     train_model()


import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import logging
import joblib
import sys
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
processed_data_path = "D:/My projects/Medical Insurane Predictor/data/Processed_data/processed_data.csv"
model_save_path = "D:/My projects/Medical Insurane Predictor/notebook/mlruns/models/insurance_price_predictor.pkl"

def train_model():
    try:
        # Set MLflow tracking URI and experiment name
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("insurance_predictor")  # Set the experiment name
        
        # End any existing active MLflow run
        mlflow.end_run()  # Ensure any previous run is ended

        # Load the preprocessed data
        df = pd.read_csv(processed_data_path)
        logging.info("Data loaded successfully.")

        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Data sample:\n{df.head()}")

        # Handle missing values
        df = df.ffill()
        logging.info("Missing values handled.")

        # Encode categorical variables
        for col in ['sex', 'smoker', 'region']:
            if col in df.columns:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
        logging.info("Categorical variables encoded.")

        # Split features and target variable
        X = df.drop(columns=['charges'])
        y = df['charges']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the model
        model = LinearRegression()

        # Start MLflow experiment
        with mlflow.start_run():  # Start a new MLflow run
            # Train the model
            model.fit(X_train, y_train)
            logging.info("Model training completed.")

            # Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Model trained: RMSE = {rmse}, R2 = {r2}")

            # Log parameters, metrics, and the model with MLflow
            mlflow.log_param("model_type", "Linear Regression")
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model, "insurance_price_predictor")
            logging.info("Model logged with MLflow.")

        # Save the model as a pickle file
        joblib.dump(model, model_save_path)
        logging.info(f"Model saved at {model_save_path}")

    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    train_model()
