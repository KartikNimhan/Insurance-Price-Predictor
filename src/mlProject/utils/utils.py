import os
import sys
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.mlProject.exception import CustomException

# Function to save a model or any object to a file
def save_object(file_path, obj):
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Save the object (model, preprocessor, etc.)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        print(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

# Function to load a model or any object from a file
def load_object(file_path):
    try:
        # Load the object (model, preprocessor, etc.)
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# Function to evaluate multiple models with GridSearchCV for hyperparameter tuning
def evaluate_models(X_train, y_train, X_test, y_test, models, param_grid):
    try:
        report = {}

        # Loop through models and their parameter grids
        for model_name, model in models.items():
            print(f"Evaluating {model_name}")
            
            # Grid search for best hyperparameters
            grid_search = GridSearchCV(model, param_grid[model_name], cv=3)
            grid_search.fit(X_train, y_train)
            
            # Set the best found parameters and train the model
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate the model using MAE and MSE
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            
            # Store performance in the report
            report[model_name] = {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_mse": train_mse,
                "test_mse": test_mse
            }

        return report
    except Exception as e:
        raise CustomException(e, sys)
