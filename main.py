# import os
# import sys
# from src.mlProject.components.data_ingestion import download_data
# from src.mlProject.components.data_transformation import preprocess_data
# from src.mlProject.components.model_trainer import train_model
# from src.mlProject.logger import logging
# from src.mlProject.exception import CustomException

# def main():
#     try:
#         # Step 1: Download data
#         logging.info("Starting data ingestion...")
#         download_data()

#         # Step 2: Data preprocessing
#         logging.info("Starting data preprocessing...")
#         preprocess_data()

#         # Step 3: Train the model
#         logging.info("Starting model training...")
#         train_model()

#         logging.info("Process completed successfully.")
#     except CustomException as e:
#         logging.error(f"Custom error occurred: {str(e)}")
#         sys.exit(1)
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

import os
import sys
import mlflow
from retrying import retry
from src.mlProject.components.data_ingestion import download_data
from src.mlProject.components.data_transformation import preprocess_data
from src.mlProject.components.model_trainer import train_model
from src.mlProject.logger import logging
from src.mlProject.exception import CustomException

def setup_mlflow():
    """
    Configure MLflow tracking URI, credentials, and experiment setup.
    """
    try:
        logging.info("Setting up MLflow...")

        # MLflow Tracking URI (e.g., local server or remote server)
        mlflow.set_tracking_uri("file:///d:/My%20projects/Medical%20Insurane%20Predictor/mlruns")

        # Set up MLflow experiment
        experiment_name = "Medical Insurance Predictor"  # Customize experiment name
        mlflow.set_experiment(experiment_name)

        logging.info(f"MLflow setup completed with experiment: {experiment_name}")
    except Exception as e:
        raise CustomException(f"Error setting up MLflow: {str(e)}", sys)

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def start_mlflow_run():
    """
    Start a new MLflow run, ensuring that any active run is properly ended.
    """
    try:
        # Handle any active run
        if mlflow.active_run() is not None:
            logging.info(f"Ending active run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()  # End the currently active run

        # Start a new run
        logging.info("Starting a new MLflow run...")
        return mlflow.start_run()
    except Exception as e:
        logging.error(f"Error starting MLflow run: {str(e)}")
        raise CustomException(f"Error starting MLflow run: {str(e)}", sys)

def main():
    """
    Main function to orchestrate the end-to-end project pipeline.
    """
    try:
        # Step 0: MLflow setup
        setup_mlflow()

        # Step 1: Download data
        logging.info("Starting data ingestion...")
        download_data()

        # Step 2: Data preprocessing
        logging.info("Starting data preprocessing...")
        preprocess_data()

        # Step 3: Train the model
        logging.info("Starting model training...")
        with start_mlflow_run() as run:
            logging.info(f"Active MLflow run ID: {run.info.run_id}")
            mlflow.log_param("pipeline_stage", "model_training")
            train_model()

        logging.info("Process completed successfully.")
    except CustomException as e:
        logging.error(f"Custom error occurred: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
