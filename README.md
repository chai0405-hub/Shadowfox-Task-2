# 🚗 Car Price Prediction Web App

A Machine Learning-powered web application that predicts the **selling price of used cars** based on key features like fuel type, kilometers driven, ownership history, and more.

---

## 📌 Project Overview

This project aims to build an intelligent system that helps users estimate the **resale value of their car**. It uses a **Random Forest Regressor** trained on real-world car data and is deployed using **Streamlit** for an interactive user interface.

---

## ✨ Features

* 🔍 Predict car selling price instantly
* 📊 Based on multiple real-world features
* 🤖 Machine Learning model (Random Forest)
* 🌐 Interactive web interface using Streamlit
* ⚡ Fast and user-friendly

---

## 🧠 Machine Learning Workflow

1. **Data Collection**

   * Dataset includes car attributes like:

     * Fuel Type
     * Years of Service
     * Showroom Price
     * Previous Owners
     * Kilometers Driven
     * Seller Type
     * Transmission Type

2. **Data Preprocessing**

   * Handling missing values
   * Feature engineering (Car Age)
   * Encoding categorical variables

3. **Model Training**

   * Algorithm: Random Forest Regressor
   * Train-test split for evaluation

4. **Model Evaluation**

   * Metrics used:

     * RMSE (Root Mean Squared Error)
     * R² Score

5. **Deployment**

   * Web app built using Streamlit

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* Streamlit
* Pickle (Model Saving)

---

## 📂 Project Structure

```
car-price-prediction/
│
├── data/
│   └── car_data.csv
│
├── model/
│   └── car_model.pkl
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 🔧 Installation

```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
```

---

### ▶️ Run the Project

1. Train the model:

```bash
python train_model.py
```

2. Launch the web app:

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This project can be deployed easily using:

* Streamlit Cloud
* Render
* Heroku

### Streamlit Deployment Steps:

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Select `app.py`
5. Click **Deploy**

---

## 📸 App Preview

> Add screenshots here after deployment

---

## 📈 Example Input

| Feature           | Value     |
| ----------------- | --------- |
| Showroom Price    | 5.0 Lakhs |
| Kilometers Driven | 30000     |
| Owners            | 1         |
| Fuel Type         | Petrol    |
| Seller Type       | Dealer    |
| Transmission      | Manual    |
| Car Age           | 5 years   |

---

## 💰 Example Output

```
Estimated Selling Price: ₹ 3.45 Lakhs
```

---

## ⚠️ Important Notes

* Use **Python 3.10 or 3.11**
* Ensure dataset is placed in `/data` folder
* Model file will be generated automatically after training

---

## 🚧 Future Improvements

* 📊 Add data visualization dashboard
* 🤖 Improve model accuracy with hyperparameter tuning
* ☁️ Deploy using Docker & CI/CD
* 📱 Mobile-friendly UI

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgements

* Scikit-learn Documentation
* Streamlit Community
* Open-source datasets

---

## 👨‍💻 Author

**Chaitanya Pawar**
GitHub: https://github.com/chai0405-hub/Shadowfox-Task-2

---
