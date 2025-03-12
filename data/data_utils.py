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
    """Loads only required data from train.csv to optimize memory usage."""

    files = {
        "stores": os.path.join(data_path, "stores.csv"),
        "items": os.path.join(data_path, "items.csv"),
        "transactions": os.path.join(data_path, "transactions.csv"),
        "oil": os.path.join(data_path, "oil.csv"),
        "holidays_events": os.path.join(data_path, "holidays_events.csv"),
        "train": os.path.join(data_path, "train.csv")
    }

    for key, file_path in files.items():
        download_file(file_path, GOOGLE_DRIVE_LINKS[key])

    df_stores = pd.read_csv(files["stores"])
    df_items = pd.read_csv(files["items"])
    df_transactions = pd.read_csv(files["transactions"])
    df_oil = pd.read_csv(files["oil"])
    df_holidays = pd.read_csv(files["holidays_events"])

    # Filter stores, items, and dates
    store_ids = df_stores[df_stores['state'] == 'Guayas']['store_nbr'].unique()
    item_ids = [106716, 1158720]
    max_date = '2014-04-01'

    chunk_size = 10**6  
    filtered_chunks = []

    for chunk in pd.read_csv(files["train"], chunksize=chunk_size, usecols=['store_nbr', 'item_nbr', 'date', 'unit_sales'], parse_dates=['date']):
        chunk_filtered = chunk[
            (chunk['store_nbr'].isin(store_ids)) & 
            (chunk['item_nbr'].isin(item_ids)) & 
            (chunk['date'] < max_date)
        ]
        filtered_chunks.append(chunk_filtered)
        del chunk  

    df_filtered = pd.concat(filtered_chunks, ignore_index=True)
    del filtered_chunks  

    df_filtered = df_filtered.groupby(['store_nbr', 'item_nbr', 'date'], as_index=False)['unit_sales'].sum()
    print(df_filtered[df_filtered['unit_sales'] > 0].head(20))  # Check if any sales are greater than 0
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

    # Ensure the dataset covers a continuous date range
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

    # Merge with store and item data
    df_filled = df_filled.merge(df_stores, on='store_nbr', how='left').merge(df_items, on='item_nbr', how='left')

    # Encode categorical features
    for col in ['city', 'state', 'type', 'family', 'class']:
        le = LabelEncoder()
        df_filled[col] = le.fit_transform(df_filled[col])

    df_filled = df_filled.sort_values(by=['store_nbr', 'item_nbr', 'date'])

    return df_filled
