import pandas as pd
import os
import gdown
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS

def download_file(file_path, url):
    """Downloads a file from Google Drive if it doesn't exist locally."""
    if not os.path.exists(file_path):
        print(f"🔽 Downloading {file_path}...")
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"✅ File exists: {file_path}")

def load_data(data_path=DATA_PATH):
    """Downloads necessary data from Google Drive and loads CSV files into DataFrames."""
    
    files = {
        "stores": os.path.join(data_path, "stores.csv"),
        "items": os.path.join(data_path, "items.csv"),
        "transactions": os.path.join(data_path, "transactions.csv"),
        "oil": os.path.join(data_path, "oil.csv"),
        "holidays_events": os.path.join(data_path, "holidays_events.csv"),
        "train": os.path.join(data_path, "train.csv"),
        "final_preprocessed_data": os.path.join(data_path, "final_preprocessed_data.csv"),
    }

    # Download files only if they don't exist
    for key, file_path in files.items():
        download_file(file_path, GOOGLE_DRIVE_LINKS[key])

    # Load DataFrames
    df_stores = pd.read_csv(files["stores"])
    df_items = pd.read_csv(files["items"])
    df_transactions = pd.read_csv(files["transactions"])
    df_oil = pd.read_csv(files["oil"])
    df_holidays = pd.read_csv(files["holidays_events"])
    final_preprocessed_data = pd.read_csv(files['final_preprocessed_data'])

    # Filter stores in 'Guayas' region
    df_stores['state'] = df_stores['state'].astype(str)
    store_ids = df_stores[df_stores['state'] == 'Guayas']['store_nbr'].unique()

    # Filter items before looping (performance optimization)
    item_families = ['GROCERY I', 'BEVERAGES', 'CLEANING']
    df_items = df_items[df_items["family"].isin(item_families)]
    item_ids = [106716, 1158720]

    # Read train.csv in chunks (memory-efficient)
    chunk_size = 10**6
    filtered_chunks = []

    for chunk in pd.read_csv(files["train"], chunksize=chunk_size, parse_dates=['date'], dtype={'onpromotion': str}):
        if "date" not in chunk.columns:
            continue  # Skip chunks without 'date'

        chunk = chunk[(chunk["store_nbr"].isin(store_ids)) & 
                      (chunk["date"] >= pd.Timestamp("2014-01-01")) & 
                      (chunk["date"] <= pd.Timestamp("2014-03-31"))]

        # Merge with df_items
        chunk = chunk.merge(df_items, on='item_nbr', how='left')

        filtered_chunks.append(chunk)

    # Combine filtered chunks
    df_train_filtered = pd.concat(filtered_chunks, ignore_index=True)

    # Ensure df_train_filtered is not empty
    if df_train_filtered.empty:
        print("⚠️ df_train_filtered is empty! Returning None.")
        return None

    # Filter selected items
    df_filtered_items = df_train_filtered[df_train_filtered["item_nbr"].isin(item_ids)].copy()

    # Ensure 'onpromotion' exists
    if "onpromotion" not in df_filtered_items.columns:
        df_filtered_items["onpromotion"] = 0

    return df_stores, df_items, df_transactions, df_oil, df_holidays, df_train_filtered, df_filtered_items, final_preprocessed_data


def preprocess_input_data(store_id, item_id, store_ids, df_stores, df_items, df_train_filtered, df_filtered_items, df_holidays, df_transactions, df_oil):
    """Preprocess input data for forecasting."""

    # Ensure df_filtered_items is not empty
    if df_filtered_items is None or df_filtered_items.empty:
        return pd.DataFrame()

    # Handle missing promotions
    df_filtered_items["onpromotion"] = df_filtered_items["onpromotion"].fillna(0).astype(int)

    # Convert dates
    df_filtered_items["date"] = pd.to_datetime(df_filtered_items["date"], errors="coerce")
    df_holidays["date"] = pd.to_datetime(df_holidays["date"], errors="coerce")
    df_transactions["date"] = pd.to_datetime(df_transactions["date"], errors="coerce")
    df_oil["date"] = pd.to_datetime(df_oil["date"], errors="coerce")

    # Feature Engineering
    df_filtered_items["lag_7"] = df_filtered_items.groupby("item_nbr")["unit_sales"].shift(7)
    df_filtered_items["rolling_mean_7"] = df_filtered_items.groupby("item_nbr")["unit_sales"].rolling(7).mean().reset_index(level=0, drop=True)

    # Merge oil prices
    df_oil = df_oil.drop_duplicates(subset=["date"])
    df_filtered_items = df_filtered_items.merge(df_oil, on="date", how="left")

    df_filtered_items["dcoilwtico"].fillna(method="ffill", inplace=True)

    # Drop 'family' if present (model was not trained with it)
    if "family" in df_filtered_items.columns:
        df_filtered_items = df_filtered_items.drop(columns=["family"])

    # 🔥 FIX: Check if 'date' exists before dropping
    if "date" in df_filtered_items.columns:
        df_filtered_items = df_filtered_items.drop(columns=["date"])

    return df_filtered_items
