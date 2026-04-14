# =========================================
# STREAMLIT APP - CAR PRICE PREDICTION
# =========================================

import streamlit as st
import numpy as np
import pickle

# =========================================
# LOAD FILES
# =========================================

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# =========================================
# HEADER
# =========================================

st.title("🚗 Car Selling Price Prediction")
st.markdown("### Predict the resale value of your car using Machine Learning")

st.write("---")

# =========================================
# USER INPUT SECTION
# =========================================

st.subheader("Enter Car Details")

present_price = st.number_input("💰 Showroom Price (Lakhs)", min_value=0.0)

kms_driven = st.number_input("📍 Kilometers Driven", min_value=0)

owner = st.selectbox("👤 Number of Previous Owners", [0, 1, 2, 3])

fuel_type = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG"])

seller_type = st.selectbox("🏪 Seller Type", ["Dealer", "Individual"])

transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

year = st.slider("📅 Year of Purchase", 2000, 2024, 2018)

# =========================================
# PREPROCESS INPUT
# =========================================

car_age = 2024 - year

fuel_encoded = encoders["fuel"].transform([fuel_type])[0]
seller_encoded = encoders["seller"].transform([seller_type])[0]
trans_encoded = encoders["trans"].transform([transmission])[0]

input_data = np.array([[
    present_price,
    kms_driven,
    owner,
    fuel_encoded,
    seller_encoded,
    trans_encoded,
    car_age
]])

# Scale input
input_scaled = scaler.transform(input_data)

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button("🔮 Predict Price"):

    prediction = model.predict(input_scaled)

    st.success(f"💵 Estimated Selling Price: ₹ {round(prediction[0], 2)} Lakhs")

# =========================================
# FOOTER
# =========================================

st.write("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")