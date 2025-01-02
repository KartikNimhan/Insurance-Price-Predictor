import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "mlProject"

# List of all essential files
list_of_files = [
    # Main directories and files
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",    # Configuration file for MLflow or any other configuration
    "params.yaml",           # Hyperparameters for your model
    "schema.yaml",           # Schema for validating data
    "main.py",               # Entry point for ML pipeline
    "app.py",                # API/Service entry point
    "requirements.txt",      # Python dependencies
    "setup.py",              # Setup script for package distribution
    "research/trials.ipynb", # Jupyter notebook for experimental setup
    "templates/index.html",  # HTML file for app/web

    # Additional directories for GitHub Actions, Docker, and Artifacts
    ".github/workflows/ci.yml",  # GitHub Actions configuration for CI/CD pipeline
    "Dockerfile",                 # Dockerfile for containerization
    "artifacts/model",            # Directory to store MLflow models and logs
    "artifacts/logs",             # Directory to store logging outputs
    "logs/mlflow.log",            # Log file for MLflow runs
]

# Creating files and directories
for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
