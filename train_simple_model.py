# train_simple_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
df = pd.read_csv("student-mat.csv")

# Select only the 6 features you use in Streamlit
features = ["studytime", "absences", "G2", "famsup", "activities", "health"]
target = "G3"

# Convert categorical to numeric manually
df["famsup"] = df["famsup"].map({"no": 0, "yes": 1})
df["activities"] = df["activities"].map({"no": 0, "yes": 1})

# Drop rows with missing values
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target]

# Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model + scaler
with open("student_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Simplified model trained and saved as student_model.pkl")