from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import os
import pickle
import sqlite3
import numpy as np
import cv2
import smtplib
from email.mime.text import MIMEText
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import speech_recognition as sr
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# Inside send_email() function:
from_email = os.getenv("EMAIL_USER")
password = os.getenv("EMAIL_PASS")

# Initialize SQLite database
# Initialize SQLite database
DB_FILE = "appointments.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table with time_slot if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        phone TEXT,
        disease TEXT,
        doctor TEXT,
        date TEXT,
        time_slot TEXT  -- Added time_slot column
    )''')

    # Check if time_slot column exists
    cursor.execute("PRAGMA table_info(appointments)")
    columns = [column[1] for column in cursor.fetchall()]
    if "time_slot" not in columns:
        cursor.execute("ALTER TABLE appointments ADD COLUMN time_slot TEXT;")
    
    conn.commit()
    conn.close()

init_db()  # Call the function to initialize the database

# Load Dataset for KYD
DATASET_PATH = os.path.join("kyd", "symptoms_diseases.csv")
df = pd.read_csv(DATASET_PATH) if os.path.exists(DATASET_PATH) else None

# Load trained models
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))  # Diabetes model
alzheimer_model = load_model("models/alzheimers_model.h5")  # Alzheimer's model
liver_model = pickle.load(open("models/liver_model.pkl", "rb"))

# Load COVID-19 model
covid_model = load_model('models/covid_model.h5')
covid_classes = ['COVID-19', 'Normal', 'Viral Pneumonia']


# Doctors data
DOCTORS = {
    "Flu": [
        {"name": "Dr. Suresh Patil", "specialty": "General Physician", "location": "Pune", "contact": "9876543201"},
        {"name": "Dr. Anjali Nair", "specialty": "General Physician", "location": "Delhi", "contact": "9876543202"}
    ],
    "Malaria": [
        {"name": "Dr. Rajiv Kapoor", "specialty": "Infectious Disease Specialist", "location": "Mumbai", "contact": "9876543203"},
        {"name": "Dr. Meena Desai", "specialty": "General Physician", "location": "Bangalore", "contact": "9876543204"}
    ],
    "COVID-19": [
        {"name": "Dr. Sanjay Verma", "specialty": "Pulmonologist", "location": "Hyderabad", "contact": "9876543205"},
        {"name": "Dr. Kavita Joshi", "specialty": "Infectious Disease Specialist", "location": "Chennai", "contact": "9876543206"}
    ],
    "Dengue": [
        {"name": "Dr. Arvind Singh", "specialty": "General Physician", "location": "Kolkata", "contact": "9876543207"},
        {"name": "Dr. Priya Sharma", "specialty": "Hematologist", "location": "Ahmedabad", "contact": "9876543208"}
    ],
    "Diabetes": [
        {"name": "Dr. Rajesh Sharma", "specialty": "Endocrinologist", "location": "Mumbai", "contact": "9876543209"},
        {"name": "Dr. Pooja Mehta", "specialty": "Diabetologist", "location": "Delhi", "contact": "9876543210"}
    ],
    "Alzheimer's": [
        {"name": "Dr. Anil Verma", "specialty": "Neurologist", "location": "Bangalore", "contact": "9876543211"},
        {"name": "Dr. Kavita Singh", "specialty": "Geriatrician", "location": "Hyderabad", "contact": "9876543212"}
    ]
}


# Class labels for Alzheimer's prediction
class_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Image upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to predict disease from symptoms
def predict_disease(user_symptoms):
    if df is None:
        return {"error": "Dataset not loaded"}

    for _, row in df.iterrows():
        csv_symptoms = row["Symptoms"].lower().split(", ")
        if all(symptom in csv_symptoms for symptom in user_symptoms):
            disease = row["Disease"].strip()
            doctors = DOCTORS.get(disease, [])
            return {
                "disease": disease,
                "description": row["Description"],
                "precautions": row["Precautions"].split(", "),
                "appointment": bool(doctors),
                "doctors": doctors
            }
    return {
        "disease": "No match found",
        "description": "Please consult a doctor.",
        "precautions": [],
        "appointment": False,
        "doctors": []
    }

# Function to preprocess Alzheimer's images
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)
    return img

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/kyd")
def kyd_page():
    return render_template("kyd.html")

@app.route("/services")
def services():
    return render_template("services.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
        user_symptoms = data.get("symptoms", [])
    else:
        user_symptoms = request.form.getlist("symptoms")
    
    result = predict_disease(user_symptoms)
    
    # If HTML form, render template
    if not request.is_json:
        return render_template("kyd.html", result=result)

    # If API call, return JSON
    return jsonify(result)

@app.route("/voice", methods=["POST"])
def voice():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening for symptoms...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            
        text = recognizer.recognize_google(audio)
        return jsonify({"symptoms": text})
    
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the voice input"})
    
    except sr.RequestError:
        return jsonify({"error": "Speech Recognition service unavailable"})
    
    except sr.WaitTimeoutError:
        return jsonify({"error": "Listening timed out. Try again."})



@app.route("/appointment", methods=["GET", "POST"])
def appointment():
    disease = request.args.get("disease", "")
    doctors = DOCTORS.get(disease, [])

    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        doctor = request.form["doctor"]
        date = request.form["date"]
        time_slot = request.form["time_slot"]  # Capture time slot

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO appointments (name, email, phone, disease, doctor, date, time_slot) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (name, email, phone, disease, doctor, date, time_slot))
        conn.commit()
        conn.close()

        send_email(email, name, disease, doctor, date, time_slot)
        flash("Appointment booked successfully! Confirmation sent to email.", "success")
        return redirect(url_for("appointment", disease=disease))

    return render_template("appointment.html", disease=disease, doctors=doctors)


# Function to send email confirmation
def send_email(to_email, patient_name, disease, doctor, date, time_slot):
    from_email = "piyushkhandare50@gamil.com"
    password = "nkbh mrid aitw djpx"
    subject = "Appointment Confirmation"
    body = f"""
    Dear {patient_name},
    
    Your appointment for {disease} has been booked successfully.
    Doctor: {doctor}
    Date: {date}
    Time Slot: {time_slot}
    
    Thank you!
    """
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    try:
        try:
            # Try normal hostname first
            server = smtplib.SMTP("smtp.gmail.com", 587)
        except Exception as dns_error:
            print("Hostname resolution failed, trying direct IP. DNS Error:", dns_error)
            server = smtplib.SMTP("74.125.133.108", 587)  # IP fallback

        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")


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

@app.route("/bmi", methods=["GET", "POST"])
def bmi_calculator():
    bmi = None
    category = ""
    if request.method == "POST":
        try:
            weight = float(request.form["weight"])
            height = float(request.form["height"]) / 100
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

@app.route("/alzheimer", methods=["GET", "POST"])
def alzheimer():
    prediction = None
    image_filename = None
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            image_filename = file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            file.save(file_path)
            img = preprocess_image(file_path)
            prediction_array = alzheimer_model.predict(img)
            predicted_class = class_labels[np.argmax(prediction_array)]
            return render_template("alzheimer.html", image=image_filename, prediction=predicted_class)
    return render_template("alzheimer.html", image=None, prediction=None)

@app.route("/liver", methods=["GET", "POST"])
def predict_liver():
    result = None
    if request.method == "POST":
        try:
            # Convert form values to float
            inputs = [float(x) for x in request.form.values()]
            final_input = np.array([inputs])  # Proper 2D shape for prediction
            
            print("Inputs received:", inputs)
            print("Input shape:", final_input.shape)

            # Make prediction
            prediction = liver_model.predict(final_input)
            result = "🚨 Liver Disease Detected 🚨" if prediction[0] == 1 else "✅ No Liver Disease Detected ✅"
        except Exception as e:
            result = f"❌ Error during prediction: {e}"
    return render_template("liver.html", prediction=result)

@app.route("/covid", methods=["GET", "POST"])
def covid_prediction():
    result = None
    image_filename = None
    if request.method == "POST":
        if "xray" not in request.files:
            return "No file part"
        file = request.files["xray"]
        if file.filename == "":
            return "No selected file"
        if file:
            image_filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], image_filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = covid_model.predict(img_array)
            result = covid_classes[np.argmax(prediction)]

            return render_template("covid.html", image=image_filename, result=result)
    return render_template("covid.html", image=None, result=None)


if __name__ == "__main__":
    app.run(debug=False)
