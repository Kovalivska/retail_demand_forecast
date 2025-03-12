# app/config.py

# Directory paths for data and model files
DATA_PATH = "/Users/svitlanakovalivska/retail_demand_forecast/data/"  # Path to the directory containing the raw data files
MODEL_PATH = '/Users/svitlanakovalivska/retail_demand_forecast/model/'  # Path to the directory containing the model files

# Google Drive file IDs for each dataset
# Replace these with actual file IDs from Google Drive where the datasets are stored
your_file_id_for_stores_csv = '1-EnanqyOH8yaKS7du9XmbcNZ4HC8zqnB'  # ID for stores data CSV
your_file_id_for_items_csv = '1-LrksOkqJsmbJLcAfE9FybHkpr7DN7_X'  # ID for items data CSV
your_file_id_for_transactions_csv = '1-VWs53l24X8CWineAXlatafRUGa5LOdJ'  # ID for transactions data CSV
your_file_id_for_oil_csv = '1-KaFDDcvcvrlrqX2N0M2Y37yDClqlKH9'  # ID for oil prices data CSV
your_file_id_for_holidays_csv = '1-OoYQpV0RYXrpg5lRvNN0Q6p8dhIlA9k'  # ID for holidays data CSV
your_file_id_for_train_csv = '1-POqe6Yxq5ooV4z4-AIrkqLm9SjFJ2VG'  # ID for training data CSV


# Google Drive links for each dataset
# These links are dynamically constructed using the file IDs, making it easy to download the data
GOOGLE_DRIVE_LINKS = {
    "stores": f"https://drive.google.com/uc?id={your_file_id_for_stores_csv}",  # Link for stores data
    "items": f"https://drive.google.com/uc?id={your_file_id_for_items_csv}",  # Link for items data
    "transactions": f"https://drive.google.com/uc?id={your_file_id_for_transactions_csv}",  # Link for transactions data
    "oil": f"https://drive.google.com/uc?id={your_file_id_for_oil_csv}",  # Link for oil prices data
    "holidays_events": f"https://drive.google.com/uc?id={your_file_id_for_holidays_csv}",  # Link for holidays data
    "train": f"https://drive.google.com/uc?id={your_file_id_for_train_csv}" # Link for training data
    
}

# Google Drive link for the model
# Replace the file ID below with the actual file ID for the XGBoost model saved in Google Drive
your_file_id_for_xgboost_model_xgb = "803893167c9b4a059f9a6e6a2e8f5503"  # ID for the XGBoost model file

# Google Drive link for the model file
GOOGLE_DRIVE_LINKS_MODELS = {
    "xgboost_model": f"https://drive.google.com/uc?id={your_file_id_for_xgboost_model_xgb}"  # Link for the XGBoost model
}