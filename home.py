# Home.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

with open("model/bank_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🏦 Loan Approval Prediction App")

st.markdown("Enter customer details to predict loan approval category:")

def user_input():
    # Example: collecting only a few features for simplicity
    Tot_Active_TL = st.number_input("Total Active Trade Lines", 0)
    Total_TL_opened_L6M = st.number_input("Total TL Opened Last 6 Months", 0)
    Tot_TL_closed_L6M = st.number_input("Total TL Closed Last 6 Months", 0)
    NETMONTHLYINCOME = st.number_input("Net Monthly Income", 0)
    AGE = st.number_input("Age", 18, 100)
    Credit_Score = st.slider("Credit Score", 300, 900)

    # Gender and marital status
    GENDER = st.selectbox("Gender", ["M", "F"])
    MARITAL = st.selectbox("Marital Status", ["Single", "Married"])

    input_dict = {
        "Tot_Active_TL": Tot_Active_TL,
        "Total_TL_opened_L6M": Total_TL_opened_L6M,
        "Tot_TL_closed_L6M": Tot_TL_closed_L6M,
        "NETMONTHLYINCOME": NETMONTHLYINCOME,
        "AGE": AGE,
        "Credit_Score": Credit_Score,
        "GENDER_F": int(GENDER == "F"),
        "GENDER_M": int(GENDER == "M"),
        "MARITALSTATUS_Single": int(MARITAL == "Single"),
        "MARITALSTATUS_Married": int(MARITAL == "Married"),
    }
    return pd.DataFrame([input_dict])

input_df = user_input()

if st.button("📊 Predict"):
    preds = model.predict_proba(input_df)[0]
    classes = ["P1 (Safe)", "P2", "P3", "P4 (Dangerous)"]
    pred_class = np.argmax(preds)
    
    st.success(f"Prediction: **{classes[pred_class]}**")
    st.progress(int(preds[pred_class] * 100))
    
    st.markdown("### 🔍 Probability Scores")
    for i, prob in enumerate(preds):
        st.write(f"{classes[i]}: {prob:.2f}")
    
    st.balloons()
