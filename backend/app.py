# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
sales_api = Flask("SuperKartRevenue Predictor")

# Load the trained machine learning model
model = joblib.load("SuperKart_revenue_forecast_model_v1_0.joblib")

# Define a route for the home page (GET request)
@sales_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Revenue Prediction API!"

# Define an endpoint for single prediction (POST request)
@sales_api.post('/v1/predict')
def predict_revenue():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted revenue as a JSON response.
    """
    # Get the JSON data from the request body
    product_store_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': product_store_data['Product_Weight'],
        'Product_Sugar_Content': product_store_data['Product_Sugar_Content'],
        'Product_Allocated_Area': product_store_data['Product_Allocated_Area'],
        'Product_Type': product_store_data['Product_Type'],
        'Product_MRP': product_store_data['Product_MRP'],
        'Store_Age': product_store_data['Store_Age'],
        'Store_Size': product_store_data['Store_Size'],
        'Store_Location_City_Type': product_store_data['Store_Location_City_Type'],
        'Store_Type': product_store_data['Store_Type'],
        'Store_Id': product_store_data['Store_Id']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_revenue = model.predict(input_data)[0]

    # Convert predicted_revenue to Python float and round to 2 decimal places
    predicted_revenue = round(float(predicted_revenue), 2)

    # Return the predicted revenue
    return jsonify({'Predicted_Revenue': predicted_revenue})


# Define an endpoint for batch prediction (POST request)
@sales_api.post('/v1/predictbatch')
def predict_revenue_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product and store details for multiple entries
    and returns the predicted revenues as a JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Apply same preprocessing as in the training notebook
    # Rename columns to match the trained model's expected features
    input_data = input_data.rename(columns={
        'Product_Id_char': 'Store_Id',
        'Store_Age_Years': 'Store_Establishment_Year', # Temporarily rename for age calculation
        'Product_Type_Category': 'Product_Type'
    })

    # Feature engineering for Store_Age
    # Assuming the context of the training data where current_year was 2009
    current_year_for_age_calculation = 2009 # This value is hardcoded as per the notebook's logic
    input_data['Store_Age'] = current_year_for_age_calculation - input_data['Store_Establishment_Year']
    input_data.drop('Store_Establishment_Year', axis=1, inplace=True)

    # Define the order of columns as expected by the model's preprocessor
    model_expected_features = [
        'Product_Weight', 'Product_Sugar_Content', 'Product_Allocated_Area',
        'Product_Type', 'Product_MRP', 'Store_Age', 'Store_Size',
        'Store_Location_City_Type', 'Store_Type', 'Store_Id'
    ]
    input_data = input_data[model_expected_features]

    # Make predictions for all entries in the DataFrame
    predicted_revenues = model.predict(input_data).tolist()

    # Convert predicted_revenues to Python floats and round to 2 decimal places
    predicted_revenues = [round(float(revenue), 2) for revenue in predicted_revenues]

    # Return the predictions as a list in the JSON response
    return jsonify(predicted_revenues)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    sales_api.run(debug=True)
