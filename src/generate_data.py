# src/generate_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = DATA_DIR / "sensor_data.csv"

np.random.seed(42)

rows = 2000  # more data for better training
data = []
current_time = datetime.now()

for i in range(rows):
    data.append({
        "soil_moisture": np.random.uniform(10, 90),    # %
        "soil_temp": np.random.uniform(10, 40),        # °C
        "humidity": np.random.uniform(30, 95),         # %
        "light": np.random.uniform(200, 2000),         # arbitrary units
        "soil_ph": np.random.uniform(5.5, 8.0),        # pH
        "rain": np.random.uniform(0, 25),              # mm
        "timestamp": (current_time - timedelta(minutes=i)).isoformat()
    })

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ CSV generated at: {OUTPUT_CSV}")
print("Shape:", df.shape)
print(df.head())
