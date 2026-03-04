from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('liver_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/liver', methods=['POST'])
def predict_liver():
    inputs = [float(x) for x in request.form.values()]
    final_input = np.array[(inputs)]
    prediction = model.predict(final_input)

    result = "Liver Disease Detected" if prediction[0] == 1 else "No Liver Disease Detected"
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

