from flask import send_file

from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors

from reportlab.lib.pagesizes import letter

from reportlab.lib.styles import getSampleStyleSheet

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
# Download PDF Report
# ----------------------
@app.route('/download_report', methods=['POST'])
def download_report():

    try:

        # ----------------------
        # Get Form Data
        # ----------------------
        crop = request.form['crop']

        accuracy_value = request.form['accuracy']

        temperature = request.form['temperature']

        humidity = request.form['humidity']

        ph = request.form['ph']

        rainfall = request.form['rainfall']

        Nvalue = request.form['Nvalue']

        Pvalue = request.form['Pvalue']

        Kvalue = request.form['Kvalue']

        # ----------------------
        # Create Buffer
        # ----------------------
        buffer = io.BytesIO()

        # ----------------------
        # Create PDF
        # ----------------------
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=30
        )

        styles = getSampleStyleSheet()

        elements = []

        # ----------------------
        # Title
        # ----------------------
        title = Paragraph(
            """
            <font size=28 color='darkgreen'>
            <b><u>CROP PREDICTION REPORT</u></b>
            </font>
            """,
            styles['Title']
        )

        elements.append(title)

        elements.append(Spacer(1, 35))

        # ----------------------
        # Date
        # ----------------------
        now = datetime.now().strftime(
            "%d %B %Y | %I:%M %p"
        )

        date_para = Paragraph(
            f"""
            <para alignment='right'>
            <font size=12>
            <b>Date:</b> {now}
            </font>
            </para>
            """,
            styles['BodyText']
        )

        elements.append(date_para)

        elements.append(Spacer(1, 30))

        # ----------------------
        # Input Section Heading
        # ----------------------
        input_heading = Paragraph(
            "<font size=18 color='darkgreen'><b>INPUT PARAMETERS</b></font>",
            styles['Heading2']
        )

        elements.append(input_heading)

        elements.append(Spacer(1, 15))

        # ----------------------
        # Input Table
        # ----------------------
        table_data = [

            ["PARAMETER", "VALUE"],

            ["Temperature", temperature],

            ["Humidity", humidity],

            ["pH", ph],

            ["Rainfall", rainfall],

            ["Nitrogen (N)", Nvalue],

            ["Phosphorus (P)", Pvalue],

            ["Potassium (K)", Kvalue]

        ]

        table = Table(
            table_data,
            colWidths=[250, 200]
        )

        table.setStyle(TableStyle([

            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),

            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),

            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),

            ('FONTSIZE', (0, 0), (-1, -1), 12),

            ('GRID', (0, 0), (-1, -1), 1.2, colors.black),

            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),

            ('TOPPADDING', (0, 0), (-1, 0), 12),

            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),

            ('BOTTOMPADDING', (0, 1), (-1, -1), 10),

            ('TOPPADDING', (0, 1), (-1, -1), 10),

            ('LEFTPADDING', (0, 0), (-1, -1), 15),

            ('RIGHTPADDING', (0, 0), (-1, -1), 15)

        ]))

        elements.append(table)

        elements.append(Spacer(1, 35))

        # ----------------------
        # Prediction Result
        # ----------------------
        result_heading = Paragraph(
            """
            <font size=18 color='darkgreen'>
            <b>PREDICTION RESULT</b>
            </font>
            """,
            styles['Heading2']
        )

        elements.append(result_heading)

        elements.append(Spacer(1, 15))

        result_para = Paragraph(
            f"""
            <font size=14>
            <b>Predicted Crop:</b> {crop}
            <br/><br/>
            <b>Model Accuracy:</b> {accuracy_value}%
            </font>
            """,
            styles['BodyText']
        )

        elements.append(result_para)

        elements.append(Spacer(1, 30))#----------

        # ----------------------
        # Model Used Section
        # ----------------------
        model_heading = Paragraph(
            """
            <font size=18 color='darkgreen'>
            <b>MODEL USED</b>
            </font>
            """,
            styles['Heading2']
        )

        elements.append(model_heading)

        elements.append(Spacer(1, 12))

        model_para = Paragraph(
            """
            <font size=14>
            Random Forest Classifier
            </font>
            """,
            styles['BodyText']
        )

        elements.append(model_para)

        elements.append(Spacer(1, 40))#--------

        footer_para = Paragraph(
            """
            <para alignment='center'>
            <font size=10 color='grey'>
            This report is generated using Machine Learning based
            crop prediction system developed for educational purposes.
            </font>
            </para>
            """,
            styles['BodyText']
        )

        elements.append(footer_para)

        # ----------------------
        # Build PDF
        # ----------------------
        # ----------------------
        # Border Function
        # ----------------------
        def add_page_border(canvas, doc):

            canvas.saveState()

            canvas.setStrokeColor(colors.darkgreen)

            canvas.setLineWidth(3)

            canvas.rect(
                20,
                20,
                letter[0] - 40,
                letter[1] - 40
            )

            canvas.restoreState()

        # ----------------------
        # Build PDF
        # ----------------------
        doc.build(
            elements,
            onFirstPage=add_page_border,
            onLaterPages=add_page_border
        )

        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="Crop_Report.pdf",
            mimetype='application/pdf'
        )

    except Exception as e:

        return f"Error generating PDF: {str(e)}"

# ----------------------
# Run App
# ----------------------
if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port
    )
