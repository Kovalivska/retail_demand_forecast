import pandas as pd  # Import pandas for data manipulation and analysis
import os  # Import os to interact with the file system
import gdown  # Import gdown to download files from Google Drive
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS  # Import paths and links for data files
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder to encode categorical features

def download_file(file_path, url):
    """Downloads a file from Google Drive if it doesn't exist locally."""
    if not os.path.exists(file_path):
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"{file_path} already exists.")

def load_data(data_path=DATA_PATH):
    """Loads only required data from local files to optimize memory usage."""

    files = {
        "stores": os.path.join(data_path, "stores.csv"),
        "items": os.path.join(data_path, "items.csv"),
        "transactions": os.path.join(data_path, "transactions.csv"),
        "oil": os.path.join(data_path, "oil.csv"),
        "holidays_events": os.path.join(data_path, "holidays_events.csv")
    }

    # Check if we have preprocessed data in inputs folder
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preprocessed_file = os.path.join(project_root, "inputs", "final_preprocessed_data.csv")
    
    # Only download if files don't exist locally
    for key, file_path in files.items():
        if not os.path.exists(file_path):
            try:
                download_file(file_path, GOOGLE_DRIVE_LINKS[key])
            except Exception as e:
                print(f"Warning: Could not download {key}: {e}")
                print(f"Using local file if available: {file_path}")

    df_stores = pd.read_csv(files["stores"])
    df_items = pd.read_csv(files["items"])
    df_transactions = pd.read_csv(files["transactions"])
    df_oil = pd.read_csv(files["oil"])
    df_holidays = pd.read_csv(files["holidays_events"])

    # Use preprocessed data if available, otherwise filter from train.csv
    if os.path.exists(preprocessed_file):
        print("Loading preprocessed data...")
        df_filtered = pd.read_csv(preprocessed_file, parse_dates=['date'])
        print(f"Loaded {len(df_filtered)} records from preprocessed data")
    else:
        print("Preprocessed data not found, trying to load from train.csv...")
        train_file = os.path.join(project_root, "inputs", "filtered_train_guayas_families.csv")
        
        if os.path.exists(train_file):
            print("Loading filtered training data...")
            df_filtered = pd.read_csv(train_file, parse_dates=['date'] if 'date' in pd.read_csv(train_file, nrows=1).columns else [])
        else:
            # Fallback: create minimal sample data
            print("Creating sample data for demonstration...")
            df_filtered = pd.DataFrame({
                'store_nbr': [2, 2, 3, 3] * 10,
                'item_nbr': [106716, 1158720, 106716, 1158720] * 10,
                'date': pd.date_range('2013-01-01', periods=40, freq='D'),
                'unit_sales': [10, 15, 8, 12] * 10
            })
    
    print(df_filtered[df_filtered['unit_sales'] > 0].head(10))  # Check if any sales are greater than 0
    print("Nonzero sales count:", (df_filtered['unit_sales'] > 0).sum())

    return df_stores, df_items, df_transactions, df_oil, df_holidays, df_filtered



def preprocess_input_data(store_id, item_id, split_date, df_stores, df_items, df_filtered):
    """Preprocess input data for model prediction, ensuring store-specific filtering."""
 
    # Convert the 'date' column to datetime format
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    split_date = pd.to_datetime(split_date)

    # **Fixed: Ensure filtering includes both store_id and item_id**
    df_filtered = df_filtered[(df_filtered['store_nbr'] == store_id) & (df_filtered['item_nbr'] == item_id)]

    # If no data exists for the selected store and item, return None
    if df_filtered.empty:
        print(f"Warning: No data found for store {store_id} and item {item_id}")
        return None

    # Check if data is already preprocessed (has necessary date features)
    date_features = ['month', 'day', 'weekofyear', 'dayofweek']
    has_date_features = all(col in df_filtered.columns for col in date_features)
    
    if has_date_features:
        print("Using already preprocessed data...")
        # Data is already preprocessed, just filter to the specific date
        df_result = df_filtered[df_filtered['date'] == split_date].copy()
        
        if df_result.empty:
            print(f"No data found for date {split_date}, using closest available data...")
            # Find the closest date
            closest_date = df_filtered.loc[(df_filtered['date'] - split_date).abs().idxmin(), 'date']
            df_result = df_filtered[df_filtered['date'] == closest_date].copy()
            print(f"Using data from {closest_date}")
            
        # Map columns from preprocessed data to expected model features
        if 'rolling_mean_7' in df_result.columns and 'rolling_mean' not in df_result.columns:
            df_result['rolling_mean'] = df_result['rolling_mean_7']
        if 'rolling_std_7' in df_result.columns and 'rolling_std' not in df_result.columns:
            df_result['rolling_std'] = df_result['rolling_std_7']
            
        # Add store information if missing
        if 'city' not in df_result.columns or 'state' not in df_result.columns or 'type' not in df_result.columns or 'cluster' not in df_result.columns:
            store_info = df_stores[df_stores['store_nbr'] == store_id]
            if not store_info.empty:
                for col in ['city', 'state', 'type', 'cluster']:
                    if col not in df_result.columns:
                        df_result[col] = store_info[col].iloc[0]
    else:
        print("Preprocessing raw data...")
        # Original preprocessing logic for raw data
        min_date = df_filtered['date'].min()
        max_date = df_filtered['date'].max()
        full_date_range = pd.DataFrame({'date': pd.date_range(start=min_date, end=max_date, freq='D')})

        # Create store-item-date combinations
        store_item_combinations = df_filtered[['store_nbr', 'item_nbr']].drop_duplicates()
        all_combinations = store_item_combinations.merge(full_date_range, how='cross')

        # Merge to fill missing dates
        df_filled = all_combinations.merge(df_filtered, on=['store_nbr', 'item_nbr', 'date'], how='left')
        df_filled['unit_sales'] = df_filled['unit_sales'].fillna(0)

        # Add date-based features
        df_filled['month'] = df_filled['date'].dt.month  
        df_filled['day'] = df_filled['date'].dt.day  
        df_filled['weekofyear'] = df_filled['date'].dt.isocalendar().week  
        df_filled['dayofweek'] = df_filled['date'].dt.dayofweek  

        # Rolling and lag features (calculated per store-item)
        df_filled = df_filled.sort_values(by=['store_nbr', 'item_nbr', 'date'])
        df_filled['rolling_mean'] = df_filled.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        df_filled['rolling_std'] = df_filled.groupby(['store_nbr', 'item_nbr'])['unit_sales'].transform(lambda x: x.rolling(window=7, min_periods=1).std())

        df_filled['lag_1'] = df_filled.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(1)
        df_filled['lag_7'] = df_filled.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(7)
        df_filled['lag_30'] = df_filled.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(30)

        df_filled.dropna(inplace=True)

        # Merge with store and item data only if needed
        if 'city' not in df_filled.columns:
            df_filled = df_filled.merge(df_stores, on='store_nbr', how='left')
        if 'family' not in df_filled.columns:
            df_filled = df_filled.merge(df_items, on='item_nbr', how='left')

        # Filter to specific date
        df_result = df_filled[df_filled['date'] == split_date].copy()

    # Encode categorical features only if they exist and are not already encoded
    categorical_cols = ['city', 'state', 'type', 'family']
    for col in categorical_cols:
        if col in df_result.columns and df_result[col].dtype == 'object':
            le = LabelEncoder()
            df_result[col] = le.fit_transform(df_result[col].astype(str))

    # Ensure numeric columns
    if 'class' in df_result.columns:
        df_result['class'] = pd.to_numeric(df_result['class'], errors='coerce')
    if 'perishable' in df_result.columns:
        df_result['perishable'] = pd.to_numeric(df_result['perishable'], errors='coerce')

    # Model expects specific features in specific order
    expected_features = ['store_nbr', 'item_nbr', 'month', 'day', 'weekofyear', 'dayofweek', 
                        'rolling_mean', 'rolling_std', 'lag_1', 'lag_7', 'lag_30', 
                        'city', 'state', 'type', 'cluster', 'family', 'class', 'perishable']
    
    # Create final dataset with only expected features
    final_data = pd.DataFrame()
    
    for feature in expected_features:
        if feature in df_result.columns:
            final_data[feature] = df_result[feature]
        else:
            # Add missing features with default values
            if feature == 'cluster':
                # Get cluster from stores data if available
                try:
                    store_info = df_stores[df_stores['store_nbr'] == store_id]
                    final_data[feature] = store_info['cluster'].iloc[0] if not store_info.empty else 0
                except:
                    final_data[feature] = 0
            else:
                final_data[feature] = 0
    
    print(f"Final features for model: {list(final_data.columns)}")
    print(f"Final data shape: {final_data.shape}")
    
    return final_data
