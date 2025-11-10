# train_model_for_app.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load dataset (make sure student-mat.csv is in same folder)
df = pd.read_csv("student-mat.csv")

print("Columns available in dataset:", df.columns.tolist())  # debug info

# Safely handle missing columns
df["study_time"] = df["studytime"]
df["attendance"] = 100 - df["absences"]  # inverse = attendance %
df["previous_grade"] = df["G2"] * 5
df["family_support"] = df["famsup"].map({"no": 0, "yes": 1})
df["extracurricular"] = df["activities"].map({"no": 0, "yes": 1})

if "health" in df.columns:
    df["health_feature"] = df["health"]
else:
    print("⚠️ 'health' column missing — using default value 3.")
    df["health_feature"] = 3

X = df[["study_time", "attendance", "previous_grade", "family_support", "extracurricular", "health_feature"]]
y = df["G3"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

with open("student_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model successfully trained and saved as student_model.pkl and scaler.pkl")
