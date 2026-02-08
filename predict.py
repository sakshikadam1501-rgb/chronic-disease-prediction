import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

print("Enter Patient Details:")

age = int(input("Age: "))
steps = int(input("Steps per day: "))
meal = int(input("Meal fixed (1=Yes,0=No): "))
previous = int(input("Previous disease (1=Yes,0=No): "))
smoking = int(input("Smoking (1=Yes,0=No): "))

# Take height and weight
height = float(input("Height (in meters, e.g. 1.65): "))
weight = float(input("Weight (in kg, e.g. 55): "))

# Calculate BMI
bmi = weight / (height ** 2)

print(f"Calculated BMI: {bmi:.2f}")

# Create DataFrame
data = pd.DataFrame([[
    age, steps, meal, previous, smoking, bmi
]], columns=[
    "age", "steps_per_day", "meal_fixed",
    "previous_disease", "smoking", "bmi"
])

# Predict
result = model.predict(data)[0]

if result == 0:
    print("Low Risk")
elif result == 1:
    print("Medium Risk")
else:
    print("High Risk")
