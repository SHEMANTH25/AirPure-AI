from flask import Flask, request, render_template
try:
    import joblib
except ImportError:
    print("Warning: joblib not installed. Install with: pip install joblib")
    joblib = None
import numpy as np
from tensorflow.keras.models import load_model
from shap_utils import build_results

app = Flask(__name__)

# --------------------------------------------------
# Load Model and Required Files
# --------------------------------------------------

try:
    print("Loading AirPure AI Model...")

    model = load_model("airquality_lstm.h5")
    scaler_x = joblib.load("scaler_x.joblib")
    label_encoder = joblib.load("label_encoder.joblib")

    print("Model Loaded Successfully!")

except Exception as e:
    print("=" * 60)
    print("MODEL LOADING ERROR")
    print(e)
    print("=" * 60)
    raise


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("home.html")


# --------------------------------------------------
# Training Performance
# --------------------------------------------------

@app.route("/training_performance")
def training_performance():
    return render_template("training_performance.html")


# --------------------------------------------------
# Manual Prediction
# --------------------------------------------------

@app.route("/predict_manually", methods=["GET", "POST"])
def predict_manually():

    if request.method == "GET":
        return render_template("index.html")

    try:

        features_list = [

            float(request.form["PM2.5"]),
            float(request.form["PM10"]),
            float(request.form["NO"]),
            float(request.form["NO2"]),
            float(request.form["NOx"]),
            float(request.form["NH3"]),
            float(request.form["CO"]),
            float(request.form["SO2"]),
            float(request.form["O3"]),
            float(request.form["Benzene"]),
            float(request.form["Toluene"]),
            float(request.form["Xylene"])

        ]

    except ValueError:

        return render_template(

            "error.html",

            error="Please enter valid numeric values for all pollutants.",

            error_code=400

        ), 400

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    if any(value < 0 for value in features_list):

        return render_template(

            "error.html",

            error="Negative pollutant values are not allowed.",

            error_code=400

        ), 400

    sample = np.array([features_list])

    sample_scaled = scaler_x.transform(sample)

    sample_reshaped = sample_scaled.reshape((1, 1, 12))

    return build_results(

        model=model,

        sample_reshaped=sample_reshaped,

        sample_scaled=sample_scaled,

        scaler_x=scaler_x,

        label_encoder=label_encoder,

        get_health_advice=get_health_advice,

        render_template=render_template

    )


# --------------------------------------------------
# Health Advice
# --------------------------------------------------

def get_health_advice(bucket):

    advice = {

        "Good":
            "The Air Quality Index is excellent. It poses little or no risk to human health. Enjoy outdoor activities.",

        "Satisfactory":
            "Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exposure.",

        "Moderate":
            "People with respiratory conditions should reduce prolonged outdoor activity.",

        "Poor":
            "Everyone may begin to experience health effects. Consider limiting outdoor exposure.",

        "Very Poor":
            "Health alert. Everyone may experience more serious health effects. Stay indoors if possible.",

        "Severe":
            "Emergency conditions. Stay indoors, wear a mask if going outside, and keep windows closed."

    }

    return advice.get(

        bucket,

        "Please follow your local health guidelines."

    )


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )