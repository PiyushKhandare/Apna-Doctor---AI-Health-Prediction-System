from flask import Flask, render_template, request, jsonify
import pandas as pd
import speech_recognition as sr
import os

app = Flask(__name__)

# Load dataset safely
DATASET_PATH = "C:\\Users\\Piyush\\Desktop\\kyd\\symptoms_diseases.csv"
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)
else:
    print(f"❌ Dataset not found at {DATASET_PATH}. Please check the path.")
    df = None  # Avoid crashing the app

# Function to find disease based on symptoms
def predict_disease(user_symptoms):
    if df is None:
        return {"error": "Dataset not loaded"}

    for _, row in df.iterrows():
        csv_symptoms = row["Symptoms"].lower().split(", ")
        if all(symptom in csv_symptoms for symptom in user_symptoms):
            return {
                "disease": row["Disease"],
                "description": row["Description"],
                "precautions": row["Precautions"].split(", ")
            }
    return {"disease": "No match found", "description": "Please consult a doctor.", "precautions": []}

# Home Route
@app.route("/")
def index():
    return render_template("kyd.html")  # Ensure "kyd.html" is inside "templates/" folder

# Route to handle symptom input
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_symptoms = data.get("symptoms", [])
    result = predict_disease(user_symptoms)
    return jsonify(result)

# Route for voice recognition
@app.route("/voice", methods=["POST"])
def voice():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening for symptoms...")
            recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
            audio = recognizer.listen(source, timeout=5)  # Stops listening after 5 seconds
            
        text = recognizer.recognize_google(audio)
        return jsonify({"symptoms": text})
    
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the voice input"})
    
    except sr.RequestError:
        return jsonify({"error": "Speech Recognition service unavailable"})
    
    except sr.WaitTimeoutError:
        return jsonify({"error": "Listening timed out. Try again."})

if __name__ == "__main__":
    app.run(debug=True)
