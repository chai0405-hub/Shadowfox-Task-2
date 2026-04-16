import joblib
import pandas as pd

def load_model():
    model = joblib.load("model/model.pkl")
    encoders = joblib.load("model/encoders.pkl")
    return model, encoders

def preprocess_input(data_dict, encoders):
    df = pd.DataFrame([data_dict])

    for col in df.columns:
        if col in encoders:
            le = encoders[col]
            df[col] = le.transform(df[col])

    return df