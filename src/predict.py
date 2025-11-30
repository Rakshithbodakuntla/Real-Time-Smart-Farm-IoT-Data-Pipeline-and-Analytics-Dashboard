# src/predict.py

import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

model = joblib.load(MODEL_PATH)

def predict(soil_moisture, soil_temp, humidity, light, soil_ph, rain):
    X = np.array([[soil_moisture, soil_temp, humidity, light, soil_ph, rain]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  # probability of irrigation_needed = 1
    label = "Irrigation Needed" if pred == 1 else "No Irrigation Needed"
    return label, proba

if __name__ == "__main__":
    # Example test
    label, proba = predict(25, 30, 60, 900, 6.5, 3)
    print("Prediction:", label)
    print("Probability of irrigation needed:", round(proba, 3))
