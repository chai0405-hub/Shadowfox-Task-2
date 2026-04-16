# Loan Approval Prediction using Machine Learning

## 📌 Overview
This project predicts loan approval status using Machine Learning.

## 🚀 Features
- Data preprocessing
- Model training (Random Forest)
- REST API using Flask
- Real-time predictions

## 🛠 Tech Stack
- Python
- Scikit-learn
- Flask

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train model
python train.py

### 3. Run API
python app.py

## 📡 API Endpoint

POST /predict

Example input:
{
  "Gender": "Male",
  "Married": "Yes",
  ...
}

## 📊 Output
Approved / Rejected
