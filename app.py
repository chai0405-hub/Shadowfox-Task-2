# ==========================================================
# 🚗 CAR PRICE PREDICTION + DASHBOARD
# ==========================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import subprocess
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = "model.pkl"
COLUMNS_PATH = "columns.pkl"
DATA_PATH = "car_data.csv"
CURRENT_YEAR = 2025

st.set_page_config(page_title="Car Price Dashboard", layout="wide")

# ==========================================================
# AUTO TRAIN
# ==========================================================

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
    st.warning("⚠️ Training model...")
    subprocess.run(["python", "train_model.py"])

# ==========================================================
# LOAD FILES
# ==========================================================

model = pickle.load(open(MODEL_PATH, "rb"))
columns = pickle.load(open(COLUMNS_PATH, "rb"))

df = pd.read_csv(DATA_PATH)

# ==========================================================
# HEADER
# ==========================================================

st.title("🚗 Car Price Prediction Dashboard")
st.markdown("### Predict price + visualize insights")

# ==========================================================
# LAYOUT
# ==========================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Details")

    present_price = st.number_input("Showroom Price", 0.0)
    kms_driven = st.number_input("Kilometers Driven", 0)
    year = st.number_input("Year", 2000, 2025)
    owner = st.selectbox("Owners", [0,1,2,3])

with col2:
    st.subheader("⚙️ Specs")

    fuel = st.selectbox("Fuel", ["Petrol","Diesel"])
    seller = st.selectbox("Seller", ["Dealer","Individual"])
    transmission = st.selectbox("Transmission", ["Manual","Automatic"])

# ==========================================================
# FEATURE ENGINEERING
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

input_df = pd.DataFrame([input_dict])

for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]

# ==========================================================
# PREDICTION
# ==========================================================

if st.button("🚀 Predict Price"):

    pred = model.predict(input_df)[0]

    st.success(f"💰 Estimated Price: ₹ {round(pred,2)} Lakhs")

    # ======================================================
    # 📊 DASHBOARD SECTION
    # ======================================================

    st.subheader("📊 Insights Dashboard")

    colA, colB = st.columns(2)

    # ---------------- PRICE DISTRIBUTION ----------------
    with colA:
        st.markdown("### 📉 Selling Price Distribution")

        fig1 = plt.figure()
        plt.hist(df["Selling_Price"], bins=20)
        plt.xlabel("Price")
        plt.ylabel("Count")

        st.pyplot(fig1)

    # ---------------- FUEL TYPE ANALYSIS ----------------
    with colB:
        st.markdown("### ⛽ Fuel Type Count")

        fuel_counts = df["Fuel_Type"].value_counts()

        fig2 = plt.figure()
        plt.bar(fuel_counts.index, fuel_counts.values)

        st.pyplot(fig2)

    # ======================================================
    # 📈 ADDITIONAL ANALYSIS
    # ======================================================

    colC, colD = st.columns(2)

    # ---------------- CAR AGE VS PRICE ----------------
    with colC:
        st.markdown("### 📅 Car Age vs Price")

        df["Car_Age"] = CURRENT_YEAR - df["Year"]

        fig3 = plt.figure()
        plt.scatter(df["Car_Age"], df["Selling_Price"])
        plt.xlabel("Car Age")
        plt.ylabel("Price")

        st.pyplot(fig3)

    # ---------------- OWNERSHIP ANALYSIS ----------------
    with colD:
        st.markdown("### 👤 Ownership Impact")

        owner_avg = df.groupby("Owner")["Selling_Price"].mean()

        fig4 = plt.figure()
        plt.bar(owner_avg.index.astype(str), owner_avg.values)

        st.pyplot(fig4)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Dashboard Enabled 🚀")
