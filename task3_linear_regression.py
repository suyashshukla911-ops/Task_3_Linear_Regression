# Task 3: Linear Regression (FINAL WORKING VERSION)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------
# 1️⃣ Automatically Detect Script Directory
# ---------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "housing_data.csv")

print("Loading file from:", file_path)

df = pd.read_csv(file_path)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# ---------------------------------------------------
# 2️⃣ Handle Missing Values
# ---------------------------------------------------
df = df.fillna(df.mean(numeric_only=True))

# ---------------------------------------------------
# 3️⃣ Separate Features & Target
# ---------------------------------------------------

# Your dataset uses 'price' in lowercase
if "price" in df.columns:
    y = df["price"]
    X = df.drop(columns=["price"])
else:
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

print("\nFeatures after encoding:")
print(X.columns)

# ---------------------------------------------------
# 4️⃣ Train-Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# 5️⃣ Train Linear Regression Model
# ---------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------
# 6️⃣ Predictions
# ---------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------
# 7️⃣ Evaluation Metrics
# ---------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE :", mae)
print("MSE :", mse)
print("R²  :", r2)

# ---------------------------------------------------
# 8️⃣ Coefficients
# ---------------------------------------------------
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nModel Coefficients:")
print(coeff_df)

print("\nIntercept:", model.intercept_)

# ---------------------------------------------------
# 9️⃣ Plot Actual vs Predicted
# ---------------------------------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.savefig(os.path.join(script_dir, "actual_vs_predicted.png"))
plt.close()

print("\nPlot saved successfully ✅")

print("\nLinear Regression Completed Successfully 🚀")
