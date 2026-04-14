# ==========================================================
# 🚗 CAR PRICE PREDICTION - ADVANCED TRAINING PIPELINE
# ==========================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ==========================================================
# 📂 CONFIG
# ==========================================================

DATA_PATH = "car_data.csv"
MODEL_PATH = "model.pkl"
COLUMNS_PATH = "columns.pkl"

CURRENT_YEAR = 2025

# ==========================================================
# 🔍 LOAD DATA
# ==========================================================

def load_data():
    print("\n📂 Loading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("❌ car_data.csv not found!")

    df = pd.read_csv(DATA_PATH)

    print("✅ Dataset Loaded Successfully")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    return df

# ==========================================================
# 🔧 PREPROCESSING
# ==========================================================

def preprocess_data(df):
    print("\n🔧 Starting preprocessing...")

    df.columns = df.columns.str.strip()

    # Drop unnecessary column
    if "Car_Name" in df.columns:
        df.drop("Car_Name", axis=1, inplace=True)
        print("✔ Dropped Car_Name")

    # Feature Engineering
    if "Year" in df.columns:
        df["Car_Age"] = CURRENT_YEAR - df["Year"]
        df.drop("Year", axis=1, inplace=True)
        print("✔ Created Car_Age")

    # Check required columns
    required_cols = [
        "Selling_Price",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Missing column: {col}")

    # Encoding categorical
    df = pd.get_dummies(df, drop_first=True)

    print("✔ Encoding done")
    print("Final columns:", df.columns.tolist())

    return df

# ==========================================================
# 📊 SPLIT DATA
# ==========================================================

def split_data(df):
    print("\n📊 Splitting dataset...")

    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns

# ==========================================================
# 🤖 TRAIN MODEL
# ==========================================================

def train_model(X_train, y_train):
    print("\n🤖 Training model...")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("✔ Model trained successfully")
    return model

# ==========================================================
# 📈 EVALUATION
# ==========================================================

def evaluate_model(model, X_test, y_test):
    print("\n📈 Evaluating model...")

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"R2 Score: {round(r2,3)}")
    print(f"MAE: {round(mae,3)}")

# ==========================================================
# 💾 SAVE ARTIFACTS
# ==========================================================

def save_artifacts(model, columns):
    print("\n💾 Saving model and columns...")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(COLUMNS_PATH, "wb") as f:
        pickle.dump(columns, f)

    print("✔ model.pkl saved")
    print("✔ columns.pkl saved")

# ==========================================================
# 🚀 MAIN
# ==========================================================

def main():
    print("\n🚗 CAR PRICE MODEL TRAINING STARTED\n")

    df = load_data()
    df = preprocess_data(df)

    (X_train, X_test, y_train, y_test), columns = split_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_artifacts(model, columns)

    print("\n🎉 TRAINING COMPLETE!")

# ==========================================================
# ▶ RUN
# ==========================================================

if __name__ == "__main__":
    main()
