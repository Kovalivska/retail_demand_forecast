# Simple Streamlit App for Deployment
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

st.set_page_config(
    page_title="Retail Demand Forecast",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("### 🏪 Retail Demand Forecast")
st.markdown(
    "<h1 style='text-align: center; color: #ed281a;'>Corporación Favorita Sales Forecasting</h1>", 
    unsafe_allow_html=True
)

# Demo mode notice
st.warning("⚠️ This is a demo version running with sample data for demonstration purposes.")

def create_sample_data():
    """Create sample data for demonstration"""
    dates = pd.date_range('2013-01-01', '2014-03-31', freq='D')
    
    # Sample stores
    stores_data = {
        'store_nbr': [1, 2, 3, 4, 5],
        'city': ['Quito', 'Quito', 'Guayaquil', 'Guayaquil', 'Cuenca'],
        'state': ['Pichincha', 'Pichincha', 'Guayas', 'Guayas', 'Azuay'],
        'type': ['A', 'B', 'A', 'C', 'B'],
        'cluster': [1, 1, 2, 3, 2]
    }
    df_stores = pd.DataFrame(stores_data)
    
    # Sample items  
    items_data = {
        'item_nbr': [106716, 1158720, 200001, 300002, 400003],
        'family': ['GROCERY I', 'BEVERAGES', 'CLEANING', 'DAIRY', 'PRODUCE'],
        'class': [1, 2, 1, 3, 2],
        'perishable': [1, 0, 0, 1, 1]
    }
    df_items = pd.DataFrame(items_data)
    
    # Generate sample sales data
    np.random.seed(42)
    sample_data = []
    
    for store in [1, 2, 3]:
        for item in [106716, 1158720]:
            for date in dates[-90:]:  # Last 90 days
                base_sales = np.random.poisson(15)
                weekend_boost = 1.3 if date.weekday() >= 5 else 1.0
                sales = int(base_sales * weekend_boost)
                
                sample_data.append({
                    'date': date,
                    'store_nbr': store,
                    'item_nbr': item,
                    'unit_sales': sales,
                    'month': date.month,
                    'day': date.day,
                    'weekofyear': date.isocalendar().week,
                    'dayofweek': date.weekday(),
                    'rolling_mean': base_sales,
                    'rolling_std': 3.0,
                    'lag_1': base_sales - 1,
                    'lag_7': base_sales - 2,
                    'lag_30': base_sales - 3
                })
    
    df_train = pd.DataFrame(sample_data)
    return df_stores, df_items, df_train

def make_prediction(store_id, item_id, prediction_date):
    """Simple prediction logic for demo"""
    np.random.seed(42 + store_id + item_id)
    base_prediction = np.random.uniform(10, 25)
    
    # Add some seasonality
    if prediction_date.weekday() >= 5:  # Weekend
        base_prediction *= 1.3
    if prediction_date.month in [11, 12]:  # Holiday season
        base_prediction *= 1.2
    
    return max(0, int(base_prediction))

def main():
    st.sidebar.header("📊 Sales Prediction Parameters")
    
    # Load sample data
    df_stores, df_items, df_train = create_sample_data()
    
    # Store selection
    available_stores = df_stores['store_nbr'].tolist()
    store_id = st.sidebar.selectbox("🏪 Select Store", available_stores)
    
    # Item selection
    available_items = df_items['item_nbr'].tolist()
    item_id = st.sidebar.selectbox("📦 Select Item", available_items)
    
    # Date selection
    min_date = datetime.date(2014, 4, 1)
    max_date = datetime.date(2014, 12, 31)
    prediction_date = st.sidebar.date_input(
        "📅 Prediction Date",
        value=datetime.date(2014, 4, 15),
        min_value=min_date,
        max_value=max_date
    )
    
    # Display store and item info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏪 Store Information")
        store_info = df_stores[df_stores['store_nbr'] == store_id].iloc[0]
        st.write(f"**Store Number:** {store_id}")
        st.write(f"**City:** {store_info['city']}")
        st.write(f"**State:** {store_info['state']}")
        st.write(f"**Type:** {store_info['type']}")
        st.write(f"**Cluster:** {store_info['cluster']}")
    
    with col2:
        st.subheader("📦 Item Information")
        item_info = df_items[df_items['item_nbr'] == item_id].iloc[0]
        st.write(f"**Item Number:** {item_id}")
        st.write(f"**Family:** {item_info['family']}")
        st.write(f"**Class:** {item_info['class']}")
        st.write(f"**Perishable:** {'Yes' if item_info['perishable'] else 'No'}")
    
    # Make prediction
    if st.sidebar.button("🔮 Make Prediction", type="primary"):
        prediction = make_prediction(store_id, item_id, prediction_date)
        
        st.success(f"🎯 **Predicted Sales for {prediction_date}:** {prediction} units")
        
        # Show historical data
        historical_data = df_train[
            (df_train['store_nbr'] == store_id) & 
            (df_train['item_nbr'] == item_id)
        ].tail(30)
        
        if not historical_data.empty:
            st.subheader("📈 Recent Sales History")
            st.line_chart(historical_data.set_index('date')['unit_sales'])
            
            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Sales", f"{historical_data['unit_sales'].mean():.1f}")
            with col2:
                st.metric("Max Sales", f"{historical_data['unit_sales'].max()}")
            with col3:
                st.metric("Min Sales", f"{historical_data['unit_sales'].min()}")
            with col4:
                st.metric("Std Dev", f"{historical_data['unit_sales'].std():.1f}")
    
    # Footer
    st.markdown("---")
    st.markdown("*This is a demonstration of retail demand forecasting using machine learning.*")

if __name__ == "__main__":
    main()
