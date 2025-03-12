# model/model_utils.py

import pickle  # Import pickle to handle model loading (not used in this version, but included for future use)
import xgboost as xgb  # Import XGBoost library for loading and using the XGBoost model
from app.config import MODEL_PATH, GOOGLE_DRIVE_LINKS_MODELS  # Import paths and links for the model files
from data.data_utils import download_file  # Import a utility function for downloading files from Google Drive

def load_model(model_path=MODEL_PATH):
    """
    Downloads necessary data from Google Drive and loads a pre-trained model.
    
    This function checks if the model file exists locally, and if not, it downloads the model from 
    the specified Google Drive link. It then loads the model into memory using XGBoost's API.
    """
    # Define paths to model files - specifying the model file to be used
    files = {
        "xgboost_model": f"{model_path}model.xgb"  # Path to the XGBoost model file
    }

    # Download model files from Google Drive if they do not exist locally
    for key, file_path in files.items():
        # Calls the download_file function to fetch the model from Google Drive
        download_file(file_path, GOOGLE_DRIVE_LINKS_MODELS[key])
   
    # Load the pre-trained XGBoost model from the downloaded .xgb file
    # Initialize the XGBoost model (XGBRegressor is commonly used for regression tasks)
    xgboost_model = xgb.XGBRegressor() 
    # Load the saved model from the specified file path
    xgboost_model.load_model(files["xgboost_model"])
    
    # Return the loaded model so it can be used for predictions
    return xgboost_model

def predict(model, input_data):
    """
    Runs prediction on input data using the pre-trained model.
    """
    # Drop 'date' column as it is not needed for prediction
    input_data = input_data.drop(columns=['date'], errors='ignore')

    # Drop 'unit_sales' column if it exists
    input_data = input_data.drop(columns=['unit_sales'], errors='ignore')

    # Convert categorical columns to integer type to match training data
    categorical_cols = ['city', 'state', 'type', 'cluster', 'family', 'class', 'perishable']
    for col in categorical_cols:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype('int')

    print("📌 Debug: Input Data Shape for Prediction:", input_data.shape)
    print("📌 Debug: Input Data Sample:\n", input_data.head())

    if 'store_nbr' not in input_data.columns:
        print("⚠️ Warning: 'store_nbr' is missing before prediction!")

    # Run prediction
    prediction = model.predict(input_data)

    print("📌 Debug: Predictions:\n", prediction[:5])  # Print first few predictions
    return prediction
