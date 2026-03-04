from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import speech_recognition as sr
import os

app = Flask(__name__)

# Load dataset
DATASET_PATH = "C:\\Users\\Piyush\\Desktop\\kyd\\symptoms_diseases.csv"
df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else None

def predict_disease(user_symptoms):
    if df is None:
        return {"error": "Dataset not loaded"}

    for _, row in df.iterrows():
        csv_symptoms = row["Symptoms"].lower().split(", ")
        if all(symptom in csv_symptoms for symptom in user_symptoms):
            return {
                "disease": row["Disease"],
                "description": row["Description"],
                "precautions": row["Precautions"].split(", "),
                "appointment": True  # Show appointment option if disease detected
            }
    return {
        "disease": "No match found",
        "description": "Please consult a doctor.",
        "precautions": [],
        "appointment": False  # No appointment option if no disease is detected
    }

@app.route("/")
def index():
    return render_template("kyd.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_symptoms = data.get("symptoms", [])
    result = predict_disease(user_symptoms)
    return jsonify(result)

@app.route("/appointment")
def appointment():
    return render_template("appointment.html")

if __name__ == "__main__":
    app.run(debug=True)
