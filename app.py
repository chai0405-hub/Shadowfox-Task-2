# ============================================
# CAR PRICE PREDICTION - STREAMLIT APP
# ============================================

import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = "model/car_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Run train_model.py first.")
    st.stop()

model = pickle.load(open(MODEL_PATH, "rb"))

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("Car Price Prediction App")
st.write("Enter car details to estimate selling price")

# ============================================
# SIDEBAR INPUTS
# ============================================

st.sidebar.header("Input Features")

present_price = st.sidebar.number_input("Showroom Price (Lakhs)", 0.0, 50.0, 5.0)
kms_driven = st.sidebar.number_input("Kilometers Driven", 0, 500000, 30000)
owners = st.sidebar.selectbox("Number of Owners", [0, 1, 2, 3])
car_age = st.sidebar.slider("Car Age (Years)", 0, 20, 5)

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

# ============================================
# ENCODING INPUTS
# ============================================

fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}

fuel = fuel_map[fuel]
seller = seller_map[seller]
transmission = trans_map[transmission]

# ============================================
# PREDICTION
# ============================================

if st.button("Predict Price"):
    input_data = np.array([[present_price, kms_driven, owners,
                            fuel, seller, transmission, car_age]])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Selling Price: ₹ {prediction:.2f} Lakhs")

# ============================================
# VISUALIZATION SECTION
# ============================================

st.subheader("Sample Data Visualization")

DATA_PATH = "car_data.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Selling Price Distribution")
        fig1 = plt.figure()
        df["Selling_Price"].hist()
        st.pyplot(fig1)

    with col2:
        st.write("Price vs KMs Driven")
        fig2 = plt.figure()
        plt.scatter(df["Kms_Driven"], df["Selling_Price"])
        plt.xlabel("Kms Driven")
        plt.ylabel("Selling Price")
        st.pyplot(fig2)

else:
    st.warning("Dataset not found for visualization")
