import streamlit as st
import numpy as np
import joblib
import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =========================
# TITLE
# =========================
st.title("🚗 Car Price Prediction System")
st.write("Predict the selling price of your car using ML")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Enter Car Details")

year = st.sidebar.slider("Year of Purchase", 2000, 2024, 2015)
present_price = st.sidebar.slider("Present Price (Lakhs)", 0.0, 20.0, 5.0)
kms = st.sidebar.slider("Kms Driven", 0, 200000, 30000)

fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
trans = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

owner = st.sidebar.slider("Number of Owners", 0, 3, 0)

# =========================
# ENCODING
# =========================
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}

fuel = fuel_map[fuel]
seller = seller_map[seller]
trans = trans_map[trans]

# =========================
# FEATURE ENGINEERING
# =========================
current_year = datetime.datetime.now().year
age = current_year - year

price_per_km = present_price / (kms + 1)
age_price_ratio = age / (present_price + 1)

# =========================
# INPUT ARRAY
# =========================
features = np.array([[present_price, kms, fuel, seller, trans, owner, age, price_per_km, age_price_ratio]])

scaled = scaler.transform(features)

# =========================
# PREDICTION
# =========================
prediction = model.predict(scaled)[0]

# =========================
# OUTPUT
# =========================
st.subheader("💰 Predicted Selling Price")

col1, col2 = st.columns(2)

with col1:
    st.metric("Price", f"₹ {prediction:.2f} Lakhs")

with col2:
    if prediction > present_price:
        st.success("Good Resale Value 🚀")
    else:
        st.warning("Depreciated Value ⚠️")

# =========================
# INFO
# =========================
st.markdown("---")
st.info("Python 3.14 Compatible | No Pillow Used")
