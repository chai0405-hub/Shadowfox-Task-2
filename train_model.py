# ============================================
# CAR PRICE PREDICTION - FINAL PROFESSIONAL CODE
# ============================================

import pandas as pd
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
        print("Put the file in the SAME folder as this file")
        exit()

    df = pd.read_csv(file_name)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)

    return df

# ============================================
# EDA (FIXED VERSION)
# ============================================

def perform_eda(df):
    print("\nPerforming EDA...")

    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Histogram
    numeric_df.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig("plots/histograms.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("plots/correlation.png")
    plt.close()

    print("EDA completed and saved in 'plots/' folder")

# ============================================
# PREPROCESS DATA
# ============================================

def preprocess_data(df):
    print("\nPreprocessing data...")

    df = df.copy()

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Feature Engineering
    df['Car_Age'] = 2025 - df['Year']

    # Drop unnecessary columns
    if 'Car_Name' in df.columns:
        df.drop(['Car_Name'], axis=1, inplace=True)

    df.drop(['Year'], axis=1, inplace=True)

    # Encode categorical columns
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
# HYPERPARAMETER TUNING
# ============================================

def tune_model(X_train, y_train):
    print("\nTuning model...")

    model = RandomForestRegressor()

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print("Best Parameters:", random_search.best_params_)

    return random_search.best_estimator_

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

    perform_eda(df)

    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = tune_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("\nDONE! PROFESSIONAL MODEL READY")

# ============================================

if __name__ == "__main__":
    main()