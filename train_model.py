# ============================================
# CAR PRICE PREDICTION - FINAL WORKING CODE
# (NO EMOJI, NO ERRORS)
# ============================================

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# LOAD DATA
# ============================================

def load_data():
    print("\nLoading dataset...")

    file_name = "car_data.csv"

    if not os.path.exists(file_name):
        print("ERROR: car_data.csv not found!")
        print("Put the file in the same folder as this script")
        exit()

    df = pd.read_csv(file_name)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)

    return df

# ============================================
# PREPROCESS DATA
# ============================================

def preprocess_data(df):
    print("\nPreprocessing data...")

    df = df.copy()

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df['Car_Age'] = 2025 - df['Year']

    if 'Car_Name' in df.columns:
        df.drop(['Car_Name'], axis=1, inplace=True)

    df.drop(['Year'], axis=1, inplace=True)

    le = LabelEncoder()

    for col in ['Fuel_Type', 'Seller_Type', 'Transmission']:
        df[col] = le.fit_transform(df[col])

    print("Preprocessing completed")

    return df

# ============================================
# SPLIT DATA
# ============================================

def split_data(df):
    print("\nSplitting data...")

    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']

    return train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================
# TRAIN MODEL
# ============================================

def train_model(X_train, y_train):
    print("\nTraining model...")

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)

    print("Model training completed")

    return model

# ============================================
# EVALUATE MODEL
# ============================================

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

# ============================================
# SAVE MODEL
# ============================================

def save_model(model):
    print("\nSaving model...")

    if not os.path.exists("model"):
        os.makedirs("model")

    with open("model/car_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved successfully")

# ============================================
# MAIN
# ============================================

def main():
    print("\nCAR PRICE PREDICTION TRAINING\n")

    df = load_data()

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("\nDONE! Model is ready.")

# ============================================

if __name__ == "__main__":
    main()
