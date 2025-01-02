import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os


if os.path.exists("/.dockerenv"): 
    model_path = '/app/models/insurance_price_predictor.pkl'
else:
    model_path = 'D:/My projects/Medical Insurane Predictor/notebook/mlruns/models/insurance_price_predictor.pkl'


model = joblib.load(model_path)


st.set_page_config(page_title="Insurance Price Prediction", page_icon="💰", layout="wide")


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


st.markdown('<p class="title">Insurance Price Prediction</p>', unsafe_allow_html=True)


age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, key="age")
sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, key="bmi")
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1, key="children")
smoker = st.selectbox("Smoker", ["Yes", "No"], key="smoker")
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"], key="region")


sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
region_mapping = {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
region = region_mapping[region]


input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=["age", "sex", "bmi", "children", "smoker", "region"])


if st.button("Predict Insurance Charges", key="predict_btn"):
    prediction = model.predict(input_data)

    st.markdown(f'<p class="result">Predicted Insurance Charge: ₹{prediction[0]:,.2f}</p>', unsafe_allow_html=True)

    st.markdown("""
        <div class="tip-box">
            <p class="subheader">Tip:</p>
            <p>To reduce your insurance premium, consider factors such as a healthy lifestyle, non-smoking habits, and maintaining a healthy BMI.</p>
        </div>
    """, unsafe_allow_html=True)
