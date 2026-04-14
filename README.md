# 🚗 Car Price Prediction System

An end-to-end Machine Learning project that predicts the **selling price of a car** based on various features such as fuel type, years of service, showroom price, kilometers driven, ownership, seller type, and transmission.

Built using **Python, Scikit-learn, and Streamlit**, this project demonstrates a complete ML pipeline from training to deployment.

---

## 🎯 Project Objective

To develop a system that helps users estimate the **approximate resale value of a car**, enabling better decision-making for sellers and buyers.

---

## 🧠 Machine Learning Approach

- Algorithm: **Random Forest Regressor**
- Dataset: Car dataset (`car_data.csv`)
- Target Variable: `Selling_Price`

### 📊 Evaluation Metrics:
- R² Score
- Mean Absolute Error (MAE)

---

## 📊 Features Used

- Present Price (Showroom Price)
- Kilometers Driven
- Year of Purchase → Converted to Car Age
- Number of Previous Owners
- Fuel Type (Petrol / Diesel)
- Seller Type (Dealer / Individual)
- Transmission (Manual / Automatic)

---

## ⚙️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **Streamlit**

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
