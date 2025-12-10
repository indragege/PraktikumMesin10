# ============================================================
# FLASK WEB APP: KNN Customer Classification (One-Page Version)
# ============================================================

from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Load model dan scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        try:
            gender = int(request.form['gender'])
            age = int(request.form['age'])
            income = float(request.form['income'])
            spending = float(request.form['spending'])

            # Preprocessing input
            input_data = np.array([[gender, age, income, spending]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)

            result = "Premium Customer" if prediction[0] == 1 else "Regular Customer"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
