# ==========================================================
# 🚗 STREAMLIT APP - PRODUCTION READY
# ==========================================================

import streamlit as st
import numpy as np
import pickle
import os

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = "model.pkl"
COLUMNS_PATH = "columns.pkl"
CURRENT_YEAR = 2025

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# ==========================================================
# LOAD MODEL
# ==========================================================

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
    st.error("❌ Model files not found! Run train_model.py first.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))
columns = pickle.load(open(COLUMNS_PATH, "rb"))

# ==========================================================
# UI DESIGN
# ==========================================================

st.title("🚗 Car Price Prediction System")
st.markdown("### Predict your car's selling price instantly")

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("💰 Showroom Price (Lakhs)", 0.0)
    kms_driven = st.number_input("🛣️ Kilometers Driven", 0)
    year = st.number_input("📅 Year", 2000, 2025)
    owner = st.selectbox("👤 Owners", [0,1,2,3])

with col2:
    fuel = st.selectbox("⛽ Fuel Type", ["Petrol","Diesel"])
    seller = st.selectbox("🏢 Seller Type", ["Dealer","Individual"])
    transmission = st.selectbox("⚙️ Transmission", ["Manual","Automatic"])

# ==========================================================
# FEATURE PROCESSING
# ==========================================================

car_age = CURRENT_YEAR - year

input_dict = {
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Owner": owner,
    "Car_Age": car_age,
    "Fuel_Type_Diesel": 1 if fuel=="Diesel" else 0,
    "Seller_Type_Individual": 1 if seller=="Individual" else 0,
    "Transmission_Manual": 1 if transmission=="Manual" else 0
}

# Convert to dataframe
input_df = pd.DataFrame([input_dict])

# Align columns
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

# ==========================================================
# PREDICTION
# ==========================================================

if st.button("🚀 Predict Price"):
    prediction = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: ₹ {round(prediction,2)} Lakhs")
