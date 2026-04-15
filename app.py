# ============================================
# CAR PRICE PREDICTION - PREMIUM DASHBOARD
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = "model/car_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found! Run train_model.py first.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

# ============================================
# HEADER
# ============================================

st.markdown("""
# 🚗 Car Price Prediction Dashboard
### Smart ML-powered resale value estimator
""")

st.markdown("---")

# ============================================
# SIDEBAR INPUTS
# ============================================

st.sidebar.header("Enter Car Details")

present_price = st.sidebar.number_input("Showroom Price (Lakhs)", 0.0, 50.0, 5.0)
kms_driven = st.sidebar.number_input("Kilometers Driven", 0, 500000, 30000)
owners = st.sidebar.selectbox("Owners", [0, 1, 2, 3])
car_age = st.sidebar.slider("Car Age (Years)", 0, 20, 5)

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# ============================================
# ENCODING
# ============================================

fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}

fuel = fuel_map[fuel]
seller = seller_map[seller]
transmission = trans_map[transmission]

# ============================================
# PREDICTION SECTION
# ============================================

st.subheader("Prediction")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Predict Price"):
        input_data = np.array([[present_price, kms_driven, owners,
                                fuel, seller, transmission, car_age]])

        prediction = model.predict(input_data)[0]

        st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")

with col2:
    st.info("Tip: Lower mileage & newer cars get better prices")

# ============================================
# DATA VISUALIZATION
# ============================================

st.markdown("---")
st.subheader("Market Insights")

DATA_PATH = "car_data.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    col1, col2 = st.columns(2)

    # Histogram (clean)
    with col1:
        st.write("Selling Price Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["Selling_Price"])
        ax.set_xlabel("Price (Lakhs)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Scatter (clean)
    with col2:
        st.write("Price vs Distance Driven")
        fig, ax = plt.subplots()
        ax.scatter(df["Kms_Driven"], df["Selling_Price"])
        ax.set_xlabel("Kms Driven")
        ax.set_ylabel("Price (Lakhs)")
        st.pyplot(fig)

    # Additional chart (NEW)
    st.subheader("Fuel Type Impact")

    fuel_counts = df.groupby("Fuel_Type")["Selling_Price"].mean()

    fig, ax = plt.subplots()
    fuel_counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("Avg Price")
    st.pyplot(fig)

else:
    st.warning("Dataset not found for charts")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("Built with Streamlit | ML Project 🚀")
