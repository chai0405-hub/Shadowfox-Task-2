# =========================================
# CAR PRICE PREDICTION - MODEL TRAINING
# =========================================

import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================================
# LOAD DATASET
# =========================================

print("📂 Loading dataset...")
df = pd.read_csv("car_data.csv")

print("✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print(df.head())

# =========================================
# BASIC CLEANING
# =========================================

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.dropna(inplace=True)

# =========================================
# FEATURE ENGINEERING
# =========================================

# Remove car name if exists
if 'Car_Name' in df.columns:
    df.drop(['Car_Name'], axis=1, inplace=True)

# Create Car Age
current_year = 2024
df['Car_Age'] = current_year - df['Year']
df.drop(['Year'], axis=1, inplace=True)

# =========================================
# ENCODING CATEGORICAL VARIABLES
# =========================================

le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()

df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Seller_Type'] = le_seller.fit_transform(df['Seller_Type'])
df['Transmission'] = le_trans.fit_transform(df['Transmission'])

# Save encoders
encoders = {
    "fuel": le_fuel,
    "seller": le_seller,
    "trans": le_trans
}

# =========================================
# DEFINE FEATURES & TARGET
# =========================================

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

print("📊 Features:", X.columns.tolist())

# =========================================
# FEATURE SCALING
# =========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================================
# MODEL TRAINING
# =========================================

print("🚀 Training Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================
# MODEL EVALUATION
# =========================================

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📈 MODEL PERFORMANCE")
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# =========================================
# FEATURE IMPORTANCE
# =========================================

importance = model.feature_importances_
for i, col in enumerate(X.columns):
    print(f"{col}: {importance[i]:.4f}")

# =========================================
# SAVE MODEL FILES
# =========================================

print("\n💾 Saving model files...")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))

print("✅ All files saved successfully!")

# =========================================
# TEST SAMPLE PREDICTION
# =========================================

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("\n🔍 Sample Prediction:", prediction[0])