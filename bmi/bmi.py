from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model  # Import load_model for .h5 files

app = Flask(__name__)

# Load trained models
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))  # Diabetes model in .pkl format
alzheimer_model = load_model("models/alzheimers_model.h5")  # Load Alzheimer's model in .h5 format

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    result = None
    if request.method == "POST":
        try:
            features = [float(request.form[key]) for key in request.form.keys()]
            input_data = np.array([features])
            prediction = diabetes_model.predict(input_data)
            result = "🚨 Diabetic 🚨" if prediction[0] == 1 else "🎉 Not Diabetic 🎉"
        except Exception as e:
            result = f"❌ Error: {e}"

    return render_template("diabetes.html", prediction_text=result)

@app.route("/alzheimer", methods=["GET", "POST"])
def alzheimer():
    result = None
    if request.method == "POST":
        try:
            features = [float(request.form[key]) for key in request.form.keys()]
            input_data = np.array([features])
            prediction = alzheimer_model.predict(input_data)
            result = "Alzheimer Detected" if prediction[0] == 1 else "No Alzheimer"
        except:
            result = "Invalid input. Please try again."

    return render_template("alzheimer.html", result=result)

# ✅ New Route for BMI Calculator
@app.route("bmi/", methods=["GET", "POST"])
def bmi_calculator():
    bmi = None
    category = ""
    if request.method == "POST":
        try:
            weight = float(request.form["weight"])
            height = float(request.form["height"]) / 100  # Convert cm to meters
            bmi = round(weight / (height ** 2), 2)
            
            if bmi < 18.5:
                category = "Underweight 😔"
            elif 18.5 <= bmi < 24.9:
                category = "Normal weight 😊"
            elif 25 <= bmi < 29.9:
                category = "Overweight 😟"
            else:
                category = "Obese 😢"

        except Exception as e:
            category = f"❌ Error: {e}"

    return render_template("bmi.html", bmi=bmi, category=category)

if __name__ == "__main__":
    app.run(debug=True)
