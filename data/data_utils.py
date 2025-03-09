# data/data_utils.py

import pandas as pd  # Import pandas for data manipulation and analysis
import os  # Import os to interact with the file system
import gdown  # Import gdown to download files from Google Drive
from app.config import DATA_PATH, GOOGLE_DRIVE_LINKS  # Import paths and links for data files
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder to encode categorical features

def download_file(file_path, url):
    """Downloads a file from Google Drive if it doesn't exist locally."""
    # Check if the file already exists locally
    if not os.path.exists(file_path):
        # If it doesn't exist, download the file from Google Drive using gdown
        gdown.download(url, file_path, quiet=False)
    else:
        # If the file already exists, print a message
        print(f"{file_path} already exists.")

def load_data(data_path=DATA_PATH):
    """Downloads necessary data from Google Drive and loads CSV files into DataFrames."""
    
    # Define the paths for all the required data files
    files = {
        "stores": f"{data_path}stores.csv",  # Path for stores data
        "items": f"{data_path}items.csv",  # Path for items data
        "transactions": f"{data_path}transactions.csv",  # Path for transactions data
        "oil": f"{data_path}oil.csv",  # Path for oil prices data
        "holidays_events": f"{data_path}holidays_events.csv",  # Path for holidays and events data
        "train": f"{data_path}train.csv", # Path for training data
        'final_preprocessed_data': f"{data_path}final_preprocessed_data.csv"  # Path for final preprocessed data
    }

    # Download the files if they don't already exist locally
    for key, file_path in files.items():
        download_file(file_path, GOOGLE_DRIVE_LINKS[key])

    # Load each downloaded CSV file into a pandas DataFrame
    df_stores = pd.read_csv(files["stores"])  # Stores data
    df_items = pd.read_csv(files["items"])  # Items data
    df_transactions = pd.read_csv(files["transactions"])  # Transactions data
    df_oil = pd.read_csv(files["oil"])  # Oil prices data
    df_holidays = pd.read_csv(files["holidays_events"])  # Holidays and events data
    final_preprocessed_data = pd.read_csv(files['final_preprocessed_data'])  # Final preprocessed data 
    
    # Load data only for stores in 'Guayas' region
    # Step 1: Convert 'date' column to datetime format (outside the chunk loop for efficiency)
    df_stores['state'] = df_stores['state'].astype(str)  # Ensure 'state' column is of type str

    # Step 2: Get store_ids for the state 'Guayas'
    store_ids = df_stores[df_stores['state'] == 'Guayas']['store_nbr'].unique()

    # Step 3: Define the item families of interest
    item_families = ['GROCERY I', 'BEVERAGES', 'CLEANING']

    # Step 4: Initialize an empty list to hold filtered chunks
    filtered_chunks = []

    # Define chunk size (adjust based on system's memory capacity)
    chunk_size = 10 ** 6

    # Step 5: Read and filter the CSV file in chunks
    for chunk in pd.read_csv(files["train"],
                         chunksize=chunk_size,
                         parse_dates=['date'],
                         dtype={'onpromotion': str}):
        # Filter by store_nbr, date range, and item families in the chunk
        chunk = chunk[(chunk['store_nbr'].isin(store_ids)) &
                  (chunk['date'] >= '2014-01-01') &
                  (chunk['date'] <= '2014-03-31')]

        # Merge with df_items (we assume df_items is already loaded)
        chunk = chunk.merge(df_items, on='item_nbr', how='left')

        # Filter by item families of interest
        chunk = chunk[chunk['family'].isin(item_families)]

        # Append the filtered chunk to the list
        filtered_chunks.append(chunk)

        # Free memory by deleting the chunk after processing
        del chunk

    # Step 6: Concatenate all filtered chunks into a single DataFrame
    df_train_filtered = pd.concat(filtered_chunks, ignore_index=True)

    # Group by date and aggregate sales
    # Filter data for selected item_ids
    item_ids = [106716, 1158720]

    df_filtered_items = df_train_filtered[df_train_filtered["item_nbr"].isin(item_ids)]

    # Return all the loaded DataFrames
    return df_stores, df_items, df_transactions, df_oil, df_holidays, df_filtered_items, final_preprocessed_data

def preprocess_input_data(store_id, item_id, df_stores, df_items, df_filtered_items, df_holidays_events, df_transactions, df_oil):

    # Handle missing values in 'onpromotion' column by filling NaN with 0 (assuming NaN means no promotion)
    df_filtered_items["onpromotion"].fillna(0, inplace = True)

    # Remove duplicates if any
    df_filtered_items = df_filtered_items.drop_duplicates()

    # Convert 'date' column to datetime format
    df_filtered_items["date"] = pd.to_datetime(df_filtered_items["date"])
    

    # If negative values exist, replace them with NaN or 0
    df_filtered_items["unit_sales"] = df_filtered_items["unit_sales"].apply(lambda x: max(x, 0))

    # If 'family' or other categorical columns need encoding
    df_filtered_items["family"] = df_filtered_items["family"].astype("category")

    # Convert 'date' column to datetime format
    df_filtered_items["date"] = pd.to_datetime(df_filtered_items["date"])
    df_holidays_events["date"] = pd.to_datetime(df_holidays_events["date"])
    df_transactions["date"] = pd.to_datetime(df_transactions["date"])
    #Filter df_transactions for sores for store_ids in Guayas region

    store_ids = df_stores[df_stores['state'] == 'Guayas']['store_nbr'].unique()
    df_transactions_filtered = df_transactions[df_transactions["store_nbr"].isin(store_ids)]

    # Aggregate transactions at the daily level
    df_daily_transactions = df_transactions_filtered.groupby('date', as_index=False)['transactions'].sum()

    # Merge with holidays
    df_holidays_events = df_holidays_events.merge(df_daily_transactions, on="date", how="left")


    # Identify outliers (3 standard deviations above mean)
    for item_nbr in item_ids:
        df_item = df_filtered_items[df_filtered_items["item_nbr"] == item_nbr].copy()

        # Aggregate sales at the daily level
        df_item = df_item.groupby('date', as_index=False)['unit_sales'].sum()

        mean_sales = df_item["unit_sales"].mean()
        std_sales = df_item["unit_sales"].std()

        # Identify outliers
        outliers = df_item[df_item["unit_sales"] > mean_sales + 3 * std_sales]

        # Merge with promotions (fixing issue with missing 'item_nbr')
        outliers = outliers.merge(
        df_filtered_items[df_filtered_items["item_nbr"] == item_nbr][["date", "onpromotion"]],
        on="date",
        how="left"
        )

        # Merge with holidays
        outliers = outliers.merge(df_holidays_events[["date", "type"]], on="date", how="left")

        # Merge with store transactions
        outliers = outliers.merge(df_daily_transactions, on="date", how="left")
        outliers = outliers.drop_duplicates(subset=['date'])

        #  Create a Holiday Flag
    df_filtered_items["holiday_flag"] = 0
    df_filtered_items.loc[df_filtered_items["date"].isin(df_holidays_events["date"]), "holiday_flag"] = 1

    # Calculate Mean + 3 Std for Non-Holiday Outliers
    for item_nbr in item_ids:
        mean_sales = df_filtered_items[df_filtered_items["item_nbr"] == item_nbr]["unit_sales"].mean()
        std_sales = df_filtered_items[df_filtered_items["item_nbr"] == item_nbr]["unit_sales"].std()
        cap_value = mean_sales + 3 * std_sales

        # Apply Capping ONLY to Non-Holiday Outliers
        df_filtered_items.loc[
            (df_filtered_items["item_nbr"] == item_nbr) &
            (df_filtered_items["unit_sales"] > cap_value) &
            (df_filtered_items["date"].isin(outliers[outliers["type"].isna()]["date"])),
            "unit_sales"
        ] = cap_value

    # Mark Outliers in the Dataset
    df_filtered_items["outlier_flag"] = 0
    df_filtered_items.loc[df_filtered_items["date"].isin(outliers["date"]), "outlier_flag"] = 1

    #Feature engeneering

    #Lag Features (e.g., sales from the past week, month):
    df_filtered_items["lag_1"] = df_filtered_items.groupby("item_nbr")["unit_sales"].shift(1)
    df_filtered_items["lag_2"] = df_filtered_items.groupby("item_nbr")["unit_sales"].shift(2)
    df_filtered_items["lag_5"] = df_filtered_items.groupby("item_nbr")["unit_sales"].shift(7)
    df_filtered_items["lag_7"] = df_filtered_items.groupby("item_nbr")["unit_sales"].shift(7)
    df_filtered_items["lag_10"] = df_filtered_items.groupby("item_nbr")["unit_sales"].shift(30)
    df_filtered_items["lag_12"] = df_filtered_items.groupby("item_nbr")["unit_sales"].shift(90)
    df_filtered_items["lag_30"] = df_filtered_items.groupby("item_nbr")["unit_sales"].shift(30)

    #Rolling Standard Deviation (captures sales volatility):
    df_filtered_items["rolling_std_7"] = df_filtered_items.groupby("item_nbr")["unit_sales"].rolling(7).std().reset_index(level=0, drop=True)

    #Rolling mean
    df_filtered_items["rolling_mean_7"] = df_filtered_items.groupby("item_nbr")["unit_sales"].rolling(7).mean().reset_index(level=0, drop=True)

    #Expanding mean:
    df_filtered_items["expanding_mean"] = df_filtered_items.groupby("item_nbr")["unit_sales"].expanding().mean().reset_index(level=0, drop=True)

    #Merge Oil Prices Data
    # Convert 'date' column to datetime format
    df_oil["date"] = pd.to_datetime(df_oil["date"])

    # Merge oil prices with the filtered dataset
    df_filtered_items = df_filtered_items.merge(df_oil, on="date", how="left")

    # Fill missing oil prices using forward fill (assuming missing values are due to non-trading days)
    df_filtered_items["dcoilwtico"].fillna(method="ffill", inplace=True)

    df=df_filtered_items[df_filtered_items['store_nbr']==28]
    df=df[df['item_nbr']==1158720]

    # Ensure your index is a DatetimeIndex
    df_copy = df.copy()
    df_copy = df_copy.set_index(pd.DatetimeIndex(df_copy['date']))


    # Now you can extract date features
    df_copy['day_of_week'] = df_copy.index.dayofweek  # 0 = Monday, 6 = Sunday
    df_copy['month'] = df_copy.index.month
    df_copy['is_weekend'] = df_copy['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df_filtered = df_copy.copy()

    return df_filtered




