# ==========================================
# STREAMLIT APP - MODERN UI DESIGN
# ==========================================

import streamlit as st
import pickle
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS ------------------
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

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------ HEADER ------------------
st.markdown("""
<h1 style='text-align: center; color: #22c55e;'>🚗 Car Price Predictor</h1>
<p style='text-align: center; font-size:18px;'>Get an instant AI-powered estimate of your car's selling price</p>
""", unsafe_allow_html=True)

# ------------------ LAYOUT ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Car Details")

    present_price = st.number_input("💰 Showroom Price (Lakhs)", min_value=0.0)
    km_driven = st.number_input("🛣️ Kilometers Driven", min_value=0)
    year = st.number_input("📅 Year of Purchase", min_value=2000, max_value=2025)
    owners = st.selectbox("👤 Number of Owners", [0, 1, 2, 3])

with col2:
    st.subheader("⚙️ Specifications")

    fuel = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel"])
    seller = st.selectbox("🏢 Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

# ------------------ FEATURE ENGINEERING ------------------
car_age = 2025 - year
fuel_diesel = 1 if fuel == "Diesel" else 0
seller_individual = 1 if seller == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

input_data = np.array([
    present_price,
    km_driven,
    owners,
    car_age,
    fuel_diesel,
    seller_individual,
    transmission_manual
]).reshape(1, -1)

# ------------------ PREDICTION BUTTON ------------------
if st.button("🚀 Predict Price"):
    prediction = model.predict(input_data)
    price = round(prediction[0], 2)

    st.markdown(f"""
    <div style='background: #1e293b; padding: 20px; border-radius: 12px; text-align:center;'>
        <h2 style='color:#22c55e;'>Estimated Price</h2>
        <h1 style='color:white;'>₹ {price} Lakhs</h1>
    </div>
    """, unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<hr>
<p style='text-align:center;'>Built with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
