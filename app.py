import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# ==============================
# LOAD MODEL
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
encoder_path = os.path.join(BASE_DIR, "model", "encoders.pkl")

model = joblib.load(model_path)
encoders = joblib.load(encoder_path)

# ==============================
# TITLE
# ==============================
st.markdown("<h1 style='text-align: center;'>Loan Approval Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# SIDEBAR
# ==============================
st.sidebar.header("About")
st.sidebar.info(
    "This ML app predicts whether a loan will be approved based on applicant details."
)

# ==============================
# INPUT FORM
# ==============================
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Amount Term", min_value=0)
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")

# ==============================
# PREDICTION
# ==============================
if st.button("Predict Loan Status"):

    input_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    df = pd.DataFrame([input_data])

    # Encode input
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
