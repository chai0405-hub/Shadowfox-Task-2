# 💰 Loan Approval Prediction using Machine Learning

An end-to-end Machine Learning project that predicts whether a loan application will be **Approved or Rejected** based on applicant details.

---

## 🚀 Live Demo
🌐 https://shadowfox-task-2-0405.streamlit.app/

---

## 📌 Overview

Loan approval is a critical decision in the financial sector. This project leverages Machine Learning to automate and improve the loan approval process by analyzing applicant data such as income, credit history, education, and more.

---

## 🎯 Features

✔ User-friendly web interface (Streamlit)  
✔ Real-time loan prediction  
✔ Data preprocessing & cleaning  
✔ Outlier detection using IQR  
✔ Machine Learning model (Random Forest)  
✔ End-to-end pipeline (Data → Model → Deployment)

---

## 🧠 How It Works

1. Data is collected from historical loan applications  
2. Missing values are handled using statistical methods  
3. Categorical features are encoded into numerical values  
4. Outliers are removed using the IQR technique  
5. Model is trained using Random Forest Classifier  
6. Predictions are served through a Streamlit web app  

---

## 📊 Input Features

- Gender  
- Married  
- Dependents  
- Education  
- Self Employed  
- Applicant Income  
- Coapplicant Income  
- Loan Amount  
- Loan Amount Term  
- Credit History  
- Property Area  

---

## 🧪 Example Output

| Input Scenario | Prediction |
|--------------|----------|
| High income + Good credit history | ✅ Approved |
| Low income + Poor credit history | ❌ Rejected |

---

## 🛠 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Joblib  

---

## 📂 Project Structure
loan-approval-ml/
│
├── data/
│ └── loan_prediction.csv
│
├── model/
│ ├── model.pkl
│ └── encoders.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
