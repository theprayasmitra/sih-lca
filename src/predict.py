import joblib
import pandas as pd

model = joblib.load("../models/profit_predictor.pkl")

data = {}

data["Ore/Mineral"] = input("Enter Ore/Mineral: ")
data["Ore/Mineral Grade"] = float(input("Enter Ore/Mineral Grade: "))
data["Amount Mined (tons)"] = float(input("Enter Amount Mined (tons): "))
data["Composition (%)"] = float(input("Enter Composition (%): "))
data["Fossil Fuel Usage (MJ)"] = float(input("Enter Fossil Fuel Usage (MJ): "))
data["Electricity Usage (kWh)"] = float(input("Enter Electricity Usage (kWh): "))
data["Water Usage (liters)"] = float(input("Enter Water Usage (liters): "))
data["Carbon Emissions (tons CO2)"] = float(input("Enter Carbon Emissions (tons CO2): "))
data["Refining Method"] = input("Enter Refining Method: ")
data["Processing Method"] = input("Enter Processing Method: ")
data["By Product"] = input("Enter By Product: ")
data["End-of-Life"] = input("Enter End-of-Life: ")
data["Recycled Content (%)"] = float(input("Enter Recycled Content (%): "))
data["Expenses (USD)"] = float(input("Enter Expenses (USD): "))

df = pd.DataFrame([data])

prediction = model.predict(df)[0]
print(f"\nPredicted Profit (USD): {prediction:.2f}")
