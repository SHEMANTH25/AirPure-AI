from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from shap_utils import build_results

app = Flask(__name__)

# Load your trained LSTM model and artifacts
# Note: Load model only if it exists, otherwise provide a dummy for now
try:
    model = load_model('airquality_lstm.h5')
    scaler_x = joblib.load('scaler_x.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except:
    print("Model or scalers not found. Please run train_lstm.py first.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/training_performance')
def training_performance():
    return render_template('training_performance.html')

@app.route('/predict_manually', methods=['POST','GET'])
def predict_manually():
    if request.method == 'POST':
        # Extract 12 features from form
        try:
            features_list = [
                float(request.form['PM2.5']), float(request.form['PM10']),
                float(request.form['NO']), float(request.form['NO2']),
                float(request.form['NOx']), float(request.form['NH3']),
                float(request.form['CO']), float(request.form['SO2']),
                float(request.form['O3']), float(request.form['Benzene']),
                float(request.form['Toluene']), float(request.form['Xylene'])
            ]
        except ValueError:
            return render_template('error.html', error="Invalid input. Please enter numeric values for all 12 pollutants.", error_code=400), 400

        # Prepare data for prediction
        sample = np.array([features_list])
        
        # Scale and Reshape
        sample_scaled = scaler_x.transform(sample)
        sample_reshaped = sample_scaled.reshape((1, 1, 12))
        
        # Predict Classification
        return build_results(

    model=model,

    sample_reshaped=sample_reshaped,

    sample_scaled=sample_scaled,

    scaler_x=scaler_x,

    label_encoder=label_encoder,

    get_health_advice=get_health_advice,

    render_template=render_template

)
    return render_template("index.html")

def get_health_advice(bucket):
    advice = {
        'Good': 'The Air Quality Index is excellent. It poses little or no risk to human health. Enjoy your outdoor activities!',
        'Satisfactory': 'The Air Quality Index is satisfactory, but there may be a minor risk for highly sensitive individuals.',
        'Moderate': 'Moderate health risk. Sensitive individuals should consider limiting prolonged outdoor exertion.',
        'Poor': 'Health warnings of emergency conditions. Everyone may begin to experience health effects.',
        'Very Poor': 'Health alert: everyone may experience more serious health effects. Avoid outdoor activities.',
        'Severe': 'Emergency condition: the entire population is likely to be affected. Stay indoors and keep windows closed.'
    }
    return advice.get(bucket, 'Please check local health guidelines.')

if __name__ == '__main__':
    app.run(debug=True)
