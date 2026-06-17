
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and artifacts
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

# Title
st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details below to predict churn risk.")

# Input form
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
    cltv = st.number_input("Customer Lifetime Value", 0, 10000, 3000)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

st.subheader("Services")
col3, col4 = st.columns(2)

with col3:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col4:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# Predict button
if st.button("🔍 Predict Churn Risk"):
    
    # Build input dictionary
    input_dict = {
        "Tenure Months": tenure,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "CLTV": cltv,
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # One-hot encode
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Scale numeric columns
    numeric_cols = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])
    
    # Predict
    prob = model.predict_proba(input_encoded)[0][1]
    prediction = model.predict(input_encoded)[0]
    
    # Show result
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if prob >= 0.6:
        risk = "🔴 HIGH RISK"
        color = "red"
    elif prob >= 0.3:
        risk = "🟡 MEDIUM RISK"
        color = "orange"
    else:
        risk = "🟢 LOW RISK"
        color = "green"
    
    st.markdown(f"<h2 style='color:{color}'>{risk}</h2>", unsafe_allow_html=True)
    st.metric("Churn Probability", f"{prob*100:.1f}%")
    
    if prob >= 0.3:
        st.warning("⚠️ This customer is at risk of churning. Consider offering a retention incentive.")
    else:
        st.success("✅ This customer is likely to stay.")

