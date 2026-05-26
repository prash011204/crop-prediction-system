from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from dotenv import load_dotenv
import os

import numpy as np
import pickle

from flask import (
    Flask,
    render_template,
    request
)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import io
import base64
import requests
import json

# ----------------------
# Flask App
# ----------------------
app = Flask(__name__)

# ----------------------
# Load Environment Variables
# ----------------------
load_dotenv()

# ----------------------
# Flask Limiter
# ----------------------
limiter = Limiter(
    get_remote_address,
    app=app
)

# ----------------------
# Load All Artifacts
# ----------------------
with open("artifacts.pkl", "rb") as f:

    artifacts = pickle.load(f)

loaded_model = artifacts["model"]

crop_mapping = artifacts["crop_mapping"]

accuracy = artifacts["accuracy"]

# ----------------------
# Load State-District JSON
# ----------------------
with open('districts.json') as f:

    districts_json = json.load(f)

states_list = sorted(
    list(set([d['state'] for d in districts_json['districts']]))
)

# ----------------------
# Weather API Key
# ----------------------
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# ----------------------
# Home Route
# ----------------------
@app.route('/')
def home():

    return render_template('index.html')

# ----------------------
# Predict Crop Route
# ----------------------
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # ----------------------
        # Crop Prediction Inputs
        # ----------------------
        temperature = float(request.form['temperature'])

        humidity = float(request.form['humidity'])

        ph = float(request.form['ph'])

        rainfall = float(request.form['rainfall'])

        Nvalue = int(request.form['Nvalue'])

        Pvalue = int(request.form['Pvalue'])

        Kvalue = int(request.form['Kvalue'])

        # ----------------------
        # Input Data
        # ----------------------
        input_data = np.array([
            temperature,
            humidity,
            ph,
            rainfall,
            Nvalue,
            Pvalue,
            Kvalue
        ]).reshape(1, 7)

        # ----------------------
        # Prediction
        # ----------------------
        prediction = loaded_model.predict(input_data)

        crop_id = int(prediction[0])

        crop_name = crop_mapping[crop_id]

        # ----------------------
        # Confidence Score
        # ----------------------
        probabilities = loaded_model.predict_proba(input_data)

        confidence = np.max(probabilities) * 100

        # ----------------------
        # Feature Importance Graph
        # ----------------------
        feature_importance = loaded_model.feature_importances_

        plt.figure(figsize=(6, 4))

        plt.bar(
            ['Temp', 'Humidity', 'pH', 'Rainfall', 'N', 'P', 'K'],
            feature_importance,
            color='green'
        )

        plt.title("Feature Importance")

        plt.xlabel("Features")

        plt.ylabel("Importance")

        plt.tight_layout()

        img = io.BytesIO()

        plt.savefig(img, format='png')

        img.seek(0)

        plot_url = base64.b64encode(
            img.getvalue()
        ).decode()

        plt.close()

        # ----------------------
        # Render Result
        # ----------------------
        return render_template(
            'result.j2',
            crop=crop_name,
            crop_id=crop_id,
            confidence=round(confidence, 2),
            accuracy=accuracy,
            plot_url=plot_url,
            weather_data=None,
            states_list=states_list,
            districts_json=districts_json
        )

    except Exception as e:

        return f"Error occurred: {str(e)}"

# ----------------------
# Weather Fetch Function
# ----------------------
def get_weather(state, district):

    weather_data = {}

    try:

        url = (
            f"http://api.weatherapi.com/v1/current.json"
            f"?key={WEATHER_API_KEY}"
            f"&q={district},{state}&aqi=no"
        )

        response = requests.get(url).json()

        print(
            f"DEBUG: Weather API Response for "
            f"{district}, {state} ->",
            response
        )

        if response.get("current"):

            weather_data = {

                "location": (
                    f"{response['location']['name']}, "
                    f"{response['location']['region']}"
                ),

                "temperature": response["current"]["temp_c"],

                "humidity": response["current"]["humidity"],

                "pressure": response["current"]["pressure_mb"],

                "weather": response["current"]["condition"]["text"],

                "wind_speed": response["current"]["wind_kph"]
            }

        else:

            weather_data = {
                "error": "Weather data not available"
            }

    except Exception as e:

        print("DEBUG: Weather fetch failed:", e)

        weather_data = {
            "error": "Weather data not available"
        }

    return weather_data

# ----------------------
# Update Weather Route
# ----------------------
@app.route('/update_weather', methods=['POST'])
@limiter.limit("10 per minute")
def update_weather():

    state = request.form.get('state')

    district = request.form.get('district')

    weather_data = get_weather(state, district)

    return render_template(
        'weather_partial.html',
        weather_data=weather_data
    )

# ----------------------
# Run App
# ----------------------
if __name__ == '__main__':

    app.run(debug=False)