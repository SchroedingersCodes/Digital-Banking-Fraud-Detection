import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Load the Random Forest model
@st.cache_resource
def load_model():
    # Ensure this filename matches exactly what you upload to GitHub
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'random_forest_smote_model.pkl')
    
    rf_model = joblib.load(model_path)
    return rf_model

try:
    rf = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'random_forest_smote_model.pkl' is in the repository.")
    st.stop()

st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️")
st.title("🛡️ Fraud Detection System (Random Forest)")
st.markdown("Enter transaction details below to predict the likelihood of fraud.")

# Create input form layout
st.header("Transaction Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender (0=Male, 1=Female)", [0, 1])
    age = st.number_input("Age (Scaled)", value=0.0)
    state = st.number_input("State ID", min_value=0, step=1)
    city = st.number_input("City ID", min_value=0, step=1)
    acc_type = st.selectbox("Account Type", [0, 1, 2])
    amount = st.number_input("Transaction Amount (Scaled)", value=0.0)
    trans_type = st.number_input("Transaction Type ID", value=0, step=1)
    merchant_cat = st.number_input("Merchant Category ID", value=0, step=1)
    balance = st.number_input("Account Balance (Scaled)", value=0.0)

with col2:
    device = st.number_input("Device ID", value=0, step=1)
    dev_type = st.number_input("Device Type ID", value=0, step=1)
    desc = st.number_input("Description ID", value=0, step=1)
    day = st.number_input("Day (Scaled)", value=0.0)
    month = st.number_input("Month", value=0.0)
    dow = st.number_input("Day of Week (Scaled)", value=0.0)
    hour = st.number_input("Hour (Scaled)", value=0.0)
    minute = st.number_input("Minute (Scaled)", value=0.0)
    second = st.number_input("Second (Scaled)", value=0.0)

# Prepare input for prediction
features = ['Gender', 'Age', 'State', 'City', 'Account_Type', 'Transaction_Amount',
            'Transaction_Type', 'Merchant_Category', 'Account_Balance', 'Transaction_Device',
            'Device_Type', 'Transaction_Description', 'Transaction_Day', 'Transaction_Month',
            'Transaction_DayOfWeek', 'Transaction_Hour', 'Transaction_Minute', 'Transaction_Second']

input_values = [[gender, age, state, city, acc_type, amount, trans_type, merchant_cat,
                 balance, device, dev_type, desc, day, month, dow, hour, minute, second]]

input_data = pd.DataFrame(input_values, columns=features)

if st.button("Predict Fraud", type="primary"):
    prediction = rf.predict(input_data)[0]
    probability = rf.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 ALERT: Potential Fraud Detected! (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Transaction looks clean. (Probability of Fraud: {probability:.2%})")
