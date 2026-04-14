# 🚗 Car Price Prediction & Analytics Dashboard

An end-to-end Machine Learning project that predicts the **selling price of a car** and provides **interactive data analysis dashboards**.

Built using **Python, Scikit-learn, Streamlit, and Matplotlib**, this project demonstrates a complete ML pipeline from training to deployment with visualization.

---

## 🎯 Project Objective

To develop a smart system that:
- Predicts **car resale value**
- Provides **data-driven insights**
- Helps users make better selling/buying decisions

---

## 🧠 Machine Learning Model

- Algorithm: **Random Forest Regressor**
- Dataset: `car_data.csv`
- Target Variable: `Selling_Price`

### 📊 Evaluation Metrics
- R² Score
- Mean Absolute Error (MAE)

---

## 📊 Features Used

- 💰 Present Price (Showroom Price)
- 🛣️ Kilometers Driven
- 📅 Year of Purchase → Converted to Car Age
- 👤 Number of Previous Owners
- ⛽ Fuel Type (Petrol / Diesel)
- 🏢 Seller Type (Dealer / Individual)
- ⚙️ Transmission (Manual / Automatic)

---

## 📈 Dashboard Features

The app includes an **interactive analytics dashboard**:

- 📉 Selling Price Distribution (Histogram)
- ⛽ Fuel Type Analysis
- 📅 Car Age vs Price (Scatter Plot)
- 👤 Ownership Impact on Price

---

## ⚙️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **Streamlit**
- **Matplotlib**

---

## 📁 Project Structure
car-price-prediction/
│
├── app.py # Streamlit Web App
├── train_model.py # Model Training Script
├── model.pkl # Trained Model
├── columns.pkl # Feature Columns (important)
├── car_data.csv # Dataset
├── requirements.txt # Dependencies
└── README.md
