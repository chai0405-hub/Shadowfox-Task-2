# ==========================================================
# 🚗 CAR PRICE PREDICTION WEB APP (FINAL VERSION)
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import subprocess

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================

MODEL_PATH = "model.pkl"
COLUMNS_PATH = "columns.pkl"
CURRENT_YEAR = 2025

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# 🎨 CUSTOM UI STYLING
# ==========================================================

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #22c55e, #16a34a);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 🤖 AUTO TRAIN MODEL IF NOT EXISTS
# ==========================================================

def train_if_needed():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
        st.warning("⚠️ Model not found. Training now... Please wait.")

        result = subprocess.run(
            ["python", "train_model.py"],
            capture_output=True,
            text=True
        )

        st.text(result.stdout)

        if not os.path.exists(MODEL_PATH):
            st.error("❌ Training failed. Please check your dataset.")
            st.stop()

        st.success("✅ Model trained successfully!")

train_if_needed()

# ==========================================================
# 📦 LOAD MODEL & COLUMNS
# ==========================================================

model = pickle.load(open(MODEL_PATH, "rb"))
columns = pickle.load(open(COLUMNS_PATH, "rb"))

# ==========================================================
# 🖥️ HEADER
# ==========================================================

st.markdown("""
<h1 style='text-align: center; color: #22c55e;'>🚗 Car Price Prediction System</h1>
<p style='text-align: center; font-size:18px;'>Enter your car details to estimate its selling price</p>
""", unsafe_allow_html=True)

# ==========================================================
# 📊 INPUT UI
# ==========================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Car Details")

    present_price = st.number_input("💰 Showroom Price (Lakhs)", min_value=0.0)
    kms_driven = st.number_input("🛣️ Kilometers Driven", min_value=0)
    year = st.number_input("📅 Year of Purchase", min_value=2000, max_value=2025)
    owner = st.selectbox("👤 Number of Owners", [0, 1, 2, 3])

with col2:
    st.subheader("⚙️ Specifications")

    fuel = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel"])
    seller = st.selectbox("🏢 Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

# ==========================================================
# 🔧 FEATURE ENGINEERING
# ==========================================================

car_age = CURRENT_YEAR - year

input_dict = {
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Owner": owner,
    "Car_Age": car_age,
    "Fuel_Type_Diesel": 1 if fuel == "Diesel" else 0,
    "Seller_Type_Individual": 1 if seller == "Individual" else 0,
    "Transmission_Manual": 1 if transmission == "Manual" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# ==========================================================
# 🔄 ALIGN FEATURES WITH TRAINING DATA
# ==========================================================

for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

# ==========================================================
# 🚀 PREDICTION
# ==========================================================

if st.button("🚀 Predict Price"):

    try:
        prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <div style='background:#1e293b;padding:20px;border-radius:12px;text-align:center;'>
            <h2 style='color:#22c55e;'>Estimated Selling Price</h2>
            <h1 style='color:white;'>₹ {round(prediction, 2)} Lakhs</h1>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# ==========================================================
# 📌 FOOTER
# ==========================================================

st.markdown("""
<hr>
<p style='text-align:center;'>Built with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
