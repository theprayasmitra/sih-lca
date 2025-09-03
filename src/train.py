import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

RANDOM_STATE = 42

df = pd.read_csv("../data/sih-dataset1.csv", sep=",")

X = df.drop(columns=["Profit (USD)"])
y = df["Profit (USD)"]

numeric_features = [
    "Ore/Mineral Grade",
    "Amount Mined (tons)",
    "Composition (%)",
    "Fossil Fuel Usage (MJ)",
    "Electricity Usage (kWh)",
    "Water Usage (liters)",
    "Carbon Emissions (tons CO2)",
    "Recycled Content (%)",
    "Expenses (USD)"
]

categorical_features = [
    "Ore/Mineral",
    "Refining Method",
    "Processing Method",
    "By Product",
    "End-of-Life"
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        random_state=RANDOM_STATE,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/profit_predictor.pkl")
print("Model saved to ../models/profit_predictor.pkl")