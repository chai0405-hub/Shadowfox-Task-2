# ============================================
# STREAMLIT APP - CAR PRICE PREDICTION
# ============================================

import streamlit as st
import numpy as np
import pickle

# ============================================
# LOAD MODEL
# ============================================

model = pickle.load(open("model/car_model.pkl", "rb"))

# ============================================
# TITLE
# ============================================

st.title("🚗 Car Price Prediction App")

st.write("Enter car details to predict selling price")

# ============================================
# USER INPUTS
# ============================================

present_price = st.number_input("Showroom Price (in Lakhs)", 0.0, 50.0)
kms_driven = st.number_input("Kilometers Driven", 0, 500000)
owners = st.selectbox("Number of Owners", [0, 1, 2, 3])
car_age = st.slider("Car Age (Years)", 0, 20)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# ============================================
# ENCODING INPUT
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

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs")
