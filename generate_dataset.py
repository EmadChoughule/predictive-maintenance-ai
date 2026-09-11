import numpy as np
import pandas as pd

np.random.seed(42)
n = 1200

temperature = np.clip(np.random.normal(72, 10, n), 45, 110)
vibration = np.clip(np.random.normal(4.5, 1.8, n), 0.5, 12)
pressure = np.clip(np.random.normal(101, 9, n), 70, 130)
rpm = np.clip(np.random.normal(1500, 220, n), 800, 2200)
operating_hours = np.random.randint(100, 10001, n)
load = np.clip(np.random.normal(68, 18, n), 10, 100)
maintenance_count = np.random.poisson(3, n)
machine_age = np.random.randint(1, 16, n)

# Synthetic risk function for an educational classification problem
risk = (
    0.055 * (temperature - 70)
    + 0.50 * (vibration - 4)
    + 0.025 * (pressure - 100)
    + 0.0015 * (rpm - 1450)
    + 0.00010 * (operating_hours - 5000)
    + 0.025 * (load - 65)
    + 0.10 * (machine_age - 7)
    - 0.12 * maintenance_count
    + np.random.normal(0, 1.4, n)
)

failure_probability = 1 / (1 + np.exp(-risk))
machine_failure = (failure_probability > 0.63).astype(int)

df = pd.DataFrame({
    "temperature": np.round(temperature, 2),
    "vibration": np.round(vibration, 2),
    "pressure": np.round(pressure, 2),
    "rpm": np.round(rpm, 0).astype(int),
    "operating_hours": operating_hours,
    "load_percentage": np.round(load, 2),
    "maintenance_count": maintenance_count,
    "machine_age_years": machine_age,
    "machine_failure": machine_failure
})

df.to_csv("data/predictive_maintenance.csv", index=False)
print("Created data/predictive_maintenance.csv")
print(df["machine_failure"].value_counts())
