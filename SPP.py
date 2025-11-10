# step1_load.py
import os
import pandas as pd

print("Current working directory:", os.getcwd())

# load dataset (must be in same folder)
df = pd.read_csv("student-mat.csv")
print("\nDataset head (first 5 rows):")
print(df.head())

print("\nColumns and types:")
print(df.dtypes)

print("\nNumber of rows:", len(df))
# step2_explore.py
import pandas as pd

df = pd.read_csv("student-mat.csv")

print("Columns:", df.columns.tolist())
print("\nAny missing values?")
print(df.isnull().sum())

print("\nSummary statistics (numeric columns):")
print(df.describe())
# step3_preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("student-mat.csv")

# show categorical columns
cat_cols = [c for c in df.columns if df[c].dtype == 'object']
print("Categorical columns:", cat_cols)

# label-encode categorical columns (simple method for beginners)
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# choose features and target
X = df.drop("G3", axis=1)
y = df["G3"]

print("\nFeatures (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("\nFirst 5 rows after encoding:")
print(X.head())
# step4_split_scale.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("student-mat.csv")

# encode categoricals
cat_cols = [c for c in df.columns if df[c].dtype == 'object']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop("G3", axis=1)
y = df["G3"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# optional: save scaled arrays to disk (numpy) — useful if running later steps separately
import numpy as np
np.save("X_train_scaled.npy", X_train_scaled)
np.save("X_test_scaled.npy", X_test_scaled)
np.save("y_train.npy", y_train.to_numpy())
np.save("y_test.npy", y_test.to_numpy())

print("Saved scaled train/test arrays to disk (X_train_scaled.npy, X_test_scaled.npy, y_train.npy, y_test.npy)")
# step5_linear.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# load preprocessed data saved earlier
X_train = np.load("X_train_scaled.npy")
X_test = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Linear Regression performance:")
print("MSE:", round(mse, 3))
print("R2 Score:", round(r2, 3))
# step6_random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_train = np.load("X_train_scaled.npy")
X_test = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Random Forest performance:")
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))
print("R2:", round(r2_score(y_test, y_pred), 3))
# step7_xgboost.py
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_train = np.load("X_train_scaled.npy")
X_test = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print("XGBoost performance:")
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))
print("R2:", round(r2_score(y_test, y_pred), 3))
# step8_svm.py
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

X_train = np.load("X_train_scaled.npy")
X_test = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

print("SVR performance:")
print("MSE:", round(mean_squared_error(y_test, y_pred), 3))
print("R2:", round(r2_score(y_test, y_pred), 3))
# step9_compare.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

X_train = np.load("X_train_scaled.npy")
X_test = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0),
    "SVR": SVR()
}

results = []
for name, m in models.items():
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    results.append({"Model": name, "R2": r2_score(y_test, y_pred), "MSE": mean_squared_error(y_test, y_pred)})

df_results = pd.DataFrame(results).sort_values("R2", ascending=False)
print(df_results)

# plot R2 scores
plt.figure(figsize=(8,5))
plt.bar(df_results["Model"], df_results["R2"])
plt.title("Model comparison (R2 score)")
plt.ylabel("R2 score")
plt.ylim(0, 1)
plt.show()
