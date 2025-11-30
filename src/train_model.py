# src/train_model.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = DATA_DIR / "sensor_data.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"

# Load data
df = pd.read_csv(DATA_CSV)

# 💧 Improved irrigation rule (more balanced)
# Old rule was too strict -> very few 1s
df["irrigation_need"] = (
    (df["soil_moisture"] < 40) &
    (df["soil_temp"] > 24) &
    (df["rain"] < 12)
).astype(int)

print("🔎 Target distribution (irrigation_need):")
print(df["irrigation_need"].value_counts(normalize=True).rename("ratio"))
print(df["irrigation_need"].value_counts())

# Features & target
X = df.drop(["irrigation_need", "timestamp"], axis=1)
y = df["irrigation_need"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🌲 RandomForest with class_weight to handle remaining imbalance
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📊 Model Evaluation:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")
