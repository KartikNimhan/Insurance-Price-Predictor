import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Check if the code is running in Docker by looking for a specific file or environment variable
if os.path.exists("/.dockerenv"):  # This is a common file inside Docker containers
    # Inside Docker container, use the path relative to the container
    model_path = '/app/models/insurance_price_predictor.pkl'
else:
    # Running locally, use the path on your local machine
    model_path = 'D:/My projects/Medical Insurane Predictor/notebook/mlruns/models/insurance_price_predictor.pkl'

# Load the model
model = joblib.load(model_path)

# Streamlit app interface
st.set_page_config(page_title="Insurance Price Prediction", page_icon="💰", layout="wide")

# Add some custom CSS for styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #1E88E5;
    }
    .predict-btn {
        background-color: #FF5722;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
    .result {
        font-size: 28px;
        font-weight: bold;
        color: #FF9800;
        text-align: center;
    }
    .header {
        font-size: 30px;
        color: #673AB7;
    }

    /* Adjust input field sizes */
    .stNumberInput, .stSelectbox, .stTextInput {
        width: 100%;
        max-width: 300px;
        margin: 10px auto;
    }

    /* Tip box styling */
    .tip-box {
        padding: 20px;
        background-color: #F1F8E9;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        color: #333;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown('<p class="title">Insurance Price Prediction</p>', unsafe_allow_html=True)

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, key="age")
sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, key="bmi")
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1, key="children")
smoker = st.selectbox("Smoker", ["Yes", "No"], key="smoker")
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"], key="region")

# Encoding categorical variables
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
region_mapping = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
region = region_mapping[region]

# Create input DataFrame
input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=["age", "sex", "bmi", "children", "smoker", "region"])

# Predict button
if st.button("Predict Insurance Charges", key="predict_btn"):
    # Make prediction in INR (as the model is already trained on INR data)
    prediction = model.predict(input_data)

    # Display the result in a beautiful format
    st.markdown(f'<p class="result">Predicted Insurance Charge: ₹{prediction[0]:,.2f}</p>', unsafe_allow_html=True)

    # Additional recommendation or message
    st.markdown("""
        <div class="tip-box">
            <p class="subheader">Tip:</p>
            <p>To reduce your insurance premium, consider factors such as a healthy lifestyle, non-smoking habits, and maintaining a healthy BMI.</p>
        </div>
    """, unsafe_allow_html=True)
