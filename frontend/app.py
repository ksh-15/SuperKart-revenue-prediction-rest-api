import streamlit as st
import pandas as pd
import requests

# Base URL of the flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the streamlit app
st.title("SuperKart Revenue Prediction App")
st.write("Enter product and store details to predict revenue, or upload a CSV for batch predictions.")

# --- Single Prediction Section ---
st.header("Single Prediction")

# Input fields for features
product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar'])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.068)
product_type = st.selectbox("Product Type", ['Fruits and Vegetables', 'Snack Foods', 'Frozen Foods', 'Dairy', 'Household', 'Baking Goods', 'Canned', 'Health and Hygiene', 'Meat', 'Soft Drinks', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood'])
product_mrp = st.number_input("Product MRP", min_value=0.0, value=147.03)
store_age = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=10)
store_size = st.selectbox("Store Size", ['Medium', 'High', 'Small'])
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 2', 'Tier 1', 'Tier 3'])
store_type = st.selectbox("Store Type", ['Supermarket Type2', 'Supermarket Type1', 'Departmental Store', 'Food Mart'])
store_id = st.selectbox("Store ID", ['OUT001',"OUT002","OUT003","OUT004"])

if st.button("Predict Single Revenue"):
    input_data = {
        'Product_Weight': product_weight,
        'Product_Sugar_Content': product_sugar_content,
        'Product_Allocated_Area': product_allocated_area,
        'Product_Type': product_type,
        'Product_MRP': product_mrp,
        'Store_Age': store_age,
        'Store_Size': store_size,
        'Store_Location_City_Type': store_location_city_type,
        'Store_Type': store_type
        'Store_Id': store_id
    }
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data)
        if response.status_code == 200:
            prediction = response.json().get('Predicted_Revenue')
            st.success(f"Predicted Revenue: ${prediction:,.2f}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Please ensure the backend is running.")

# --- Batch Prediction Section ---
st.header("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])

if uploaded_file is not None and st.button("Predict Batch Revenue"):
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files)
        if response.status_code == 200:
            predictions_data = response.json()
            predictions_df = pd.DataFrame(predictions_data)
            st.success("Batch predictions successful!")
            st.dataframe(predictions_df)
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Please ensure the backend is running.")
