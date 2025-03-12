import streamlit as st  # Streamlit for web interface
import matplotlib.pyplot as plt  # For visualization
import pandas as pd  # For data handling
import datetime  # For handling date inputs
from sklearn.preprocessing import LabelEncoder  # For encoding categorical features
from app.config import DATA_PATH, MODEL_PATH  # Import paths for data and model
from data.data_utils import load_data, preprocess_input_data  # Functions to load and preprocess data
from model.model_utils import load_model, predict  # Functions to load the model and make predictions

# Fixed max training date
max_train_date = datetime.date(2014, 3, 31)  # Last date in training data

# Load Logo
st.image("/Users/svitlanakovalivska/retail_demand_forecast/outputs/logo.png", width=200)

# Title with dark red color
st.markdown(
    "<h1 style='text-align: center; color: #ed281a;'>Corporación Favorita Sales Forecasting</h1>", 
    unsafe_allow_html=True
)

def generate_future_features(store_id, item_id, forecast_date, df_train, df_stores, df_items):
    """
    Generate future feature values for a given store, item, and date.
    This is used when forecast_date > max_train_date.
    """
    print(f"⚠️ Selected date {forecast_date} is beyond training data. Generating future features...")

    # Convert forecast_date to Pandas datetime
    forecast_date = pd.to_datetime(forecast_date)

    # Filter training data for this store/item
    df_filtered = df_train[(df_train['store_nbr'] == store_id) & (df_train['item_nbr'] == item_id)]
    
    if df_filtered.empty:
        print(f"⚠️ No historical data found for Store {store_id}, Item {item_id}")
        return None

    # Ensure dates are sorted
    df_filtered = df_filtered.sort_values(by="date")

    # Get last known record for feature generation
    last_known_data = df_filtered.iloc[-1].copy()

    # Initialize future data
    future_data = pd.DataFrame(columns=df_train.columns.drop("unit_sales", errors="ignore"))  

    # Extract store and item features
    store_info = df_stores[df_stores['store_nbr'] == store_id].iloc[0] if store_id in df_stores['store_nbr'].values else None
    item_info = df_items[df_items['item_nbr'] == item_id].iloc[0] if item_id in df_items['item_nbr'].values else None

    # Fill categorical/store/item-related features
    categorical_cols = ['store_nbr', 'item_nbr', 'city', 'state', 'type', 'cluster', 'family', 'class', 'perishable']
    
    for col in categorical_cols:
        if col in last_known_data:
            future_data[col] = [last_known_data[col]]
        elif col in df_stores.columns and store_info is not None:
            future_data[col] = [store_info[col]]
        elif col in df_items.columns and item_info is not None:
            future_data[col] = [item_info[col]]
        else:
            future_data[col] = ['Unknown']  # Default for unseen categories

    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in ['city', 'state', 'type', 'family']:
        le = LabelEncoder()
        if col in df_train.columns:
            le.fit(df_train[col].astype(str))
            label_encoders[col] = le
            if future_data[col][0] in le.classes_:
                future_data[col] = le.transform(future_data[col].astype(str))
            else:
                future_data[col] = -1  # Assign -1 if category is unknown
        else:
            future_data[col] = -1

    # Convert categorical columns (except store/item) to int
    for col in ['city', 'state', 'type', 'cluster', 'family', 'class']:
        future_data[col] = future_data[col].astype('int')

    # Convert `perishable` to int (replace `None` with 0)
    future_data['perishable'] = future_data['perishable'].fillna(0).astype('int')

    # Generate date-based features
    future_data["month"] = forecast_date.month
    future_data["day"] = forecast_date.day
    future_data["weekofyear"] = forecast_date.isocalendar()[1]
    future_data["dayofweek"] = forecast_date.weekday()

    # Generate rolling means and lag features
    future_data["rolling_mean"] = df_filtered['unit_sales'].rolling(window=7).mean().iloc[-1]
    future_data["rolling_std"] = df_filtered['unit_sales'].rolling(window=7).std().iloc[-1]

    # Generate lag features
    future_data["lag_1"] = df_filtered['unit_sales'].shift(1).iloc[-1]
    future_data["lag_7"] = df_filtered['unit_sales'].shift(7).iloc[-1]
    future_data["lag_30"] = df_filtered['unit_sales'].shift(30).iloc[-1]

    # add forecast date
    future_data['date'] = forecast_date

    # check for missing values and fill with 0
    future_data.fillna(0, inplace=True)

    # fix the order of columns
    expected_features = ['store_nbr', 'item_nbr', 'date', 'month', 'day', 'weekofyear', 'dayofweek', 
                         'rolling_mean', 'rolling_std', 'lag_1', 'lag_7', 'lag_30', 
                         'city', 'state', 'type', 'cluster', 'family', 'class', 'perishable']
    
    future_data = future_data[expected_features]  

    
    print(f"✅ Future features generated for {forecast_date}")
    print("📌 **Data Types:**", future_data.dtypes)
    print("📌 **Final Input Features Order:**", list(future_data.columns))

    return future_data


def main():
    # **Load Data & Model**
    df_stores, df_items, df_transactions, df_oil, df_holidays, df_train = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)

    # **Store, Item, and Date Selection**
    available_stores = df_train['store_nbr'].unique().tolist()
    store_id = st.selectbox("Store", available_stores)

    available_items = df_train['item_nbr'].unique().tolist()
    item_id = st.selectbox("Item", available_items)

    default_date = datetime.date(2014, 3, 1)
    min_date = datetime.date(2013, 1, 1)
    max_date = datetime.date(2014, 6, 7)
    forecast_date = st.date_input("Forecast Date", value=default_date, min_value=min_date, max_value=max_date)

    if st.button("Get Forecast"):
        if forecast_date > max_train_date:
            input_data = generate_future_features(store_id, item_id, forecast_date, df_train, df_stores, df_items)
        else:
            input_data = preprocess_input_data(store_id, item_id, forecast_date, df_stores, df_items, df_train)

        if input_data is not None and not input_data.empty:
            prediction = predict(model, input_data)
            if prediction.size > 0:
                
                st.write(f"📌 **Predictions for Store {store_id} on {forecast_date}: {prediction[0]:.2f}**")
                plot_predictions(store_id, item_id, forecast_date, df_train, prediction[0])

    # Add GitHub and data source links
    st.markdown("""
    **📂 Data Source:** [Corporación Favorita Grocery Sales Forecasting Dataset](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)  
    **🔗 Models and Scripts are in the Project Repository:** [GitHub - Retail Demand Forecast](https://github.com/Kovalivska/retail_demand_forecast.git)
    """)


def plot_predictions(store_id, item_id, forecast_date, df_train, prediction):
    """
    Function to visualize sales and forecasted value.
    """
    df_train['date'] = pd.to_datetime(df_train['date'])
    item_store_data = df_train[(df_train['store_nbr'] == store_id) & (df_train['item_nbr'] == item_id)]
    
    plt.figure(figsize=(12, 6))
    
    split_date = pd.to_datetime(forecast_date)
    train_series = item_store_data[item_store_data['date'] < split_date]
    test_series = item_store_data[item_store_data['date'] >= split_date]

    plt.plot(train_series['date'], train_series['unit_sales'], label='Actual Sales (Train)', color='black')
    plt.plot(test_series['date'], test_series['unit_sales'], label='Actual Sales (Future)', color='blue')
    plt.scatter([forecast_date], [prediction], color='green', s=100, label='Predicted Sales', zorder=3)
    
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Unit Sales")
    plt.title(f"Sales Forecast for Store {store_id}, Item {item_id}")
    st.pyplot(plt)


if __name__ == "__main__":
    main()
