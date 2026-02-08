from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    steps = int(request.form["steps"])
    meal = int(request.form["meal"])
    previous = int(request.form["previous"])
    smoking = int(request.form["smoking"])
    height = float(request.form["height"])
    weight = float(request.form["weight"])

    bmi = weight / (height ** 2)

    data = pd.DataFrame([[
        age, steps, meal, previous, smoking, bmi
    ]], columns=[
        "age", "steps_per_day", "meal_fixed",
        "previous_disease", "smoking", "bmi"
    ])

    result = model.predict(data)[0]

    if result == 0:
        risk = "Low Risk"
    elif result == 1:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return render_template("index.html", prediction=risk, bmi=round(bmi, 2))


if __name__ == "__main__":
    app.run(debug=True)
