# ==========================================
# CAR PRICE PREDICTION - TRAINING SCRIPT
# ==========================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ==========================================
# LOAD DATA
# ==========================================

def load_data():
    print("📂 Loading dataset...")
    df = pd.read_csv("car_data.csv")
    print("✅ Dataset Loaded Successfully")
    print("Shape:", df.shape)
    print(df.head())
    return df


# ==========================================
# PREPROCESSING
# ==========================================

def preprocess_data(df):
    print("\n🔧 Preprocessing data...")

    # Drop unnecessary column
    if 'Car_Name' in df.columns:
        df.drop(['Car_Name'], axis=1, inplace=True)

    # Feature Engineering: Car Age
    if 'Year' in df.columns:
        df['Car_Age'] = 2025 - df['Year']
        df.drop(['Year'], axis=1, inplace=True)

    # Convert categorical variables
    df = pd.get_dummies(df, drop_first=True)

    print("✅ Preprocessing Completed")
    print("Columns after encoding:", df.columns.tolist())

    return df


# ==========================================
# SPLIT DATA
# ==========================================

def split_data(df):
    print("\n📊 Splitting dataset...")

    X = df.drop(['Selling_Price'], axis=1)
    y = df['Selling_Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return X_train, X_test, y_train, y_test


# ==========================================
# TRAIN MODEL
# ==========================================

def train_model(X_train, y_train):
    print("\n🤖 Training model...")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("✅ Model Training Completed")
    return model


# ==========================================
# EVALUATE MODEL
# ==========================================

def evaluate_model(model, X_test, y_test):
    print("\n📈 Evaluating model...")

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"🔹 R2 Score: {round(r2, 3)}")
    print(f"🔹 Mean Absolute Error: {round(mae, 3)}")


# ==========================================
# SAVE MODEL
# ==========================================

def save_model(model):
    print("\n💾 Saving model...")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved as model.pkl")


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():
    print("🚗 CAR PRICE PREDICTION TRAINING STARTED\n")

    df = load_data()

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("\n🎉 Training Completed Successfully!")


# ==========================================
# RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    main()
