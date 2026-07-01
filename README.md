# 🌍 AirPure AI

> **AI-Powered Air Quality Prediction using LSTM Deep Learning & SHAP Explainable AI**

AirPure AI is an intelligent web application that predicts **Air Quality Index (AQI)** categories using a trained **Long Short-Term Memory (LSTM)** neural network. The application also integrates **SHAP (SHapley Additive Explanations)** to explain how each pollutant contributes to the model's prediction, making the AI transparent and interpretable.

---

# 🚀 Live Demo

🔗 **Website**

https://airpure-ai-production.up.railway.app

---

# 📸 Preview

> Add screenshots here after deployment.

| Home | Prediction | SHAP Dashboard |
|------|------------|----------------|
| Home Page | AQI Prediction | Explainable AI |

---

# ✨ Features

✅ Modern Responsive User Interface

✅ Air Quality Prediction using LSTM

✅ Explainable AI using SHAP

✅ Prediction Confidence Score

✅ Top 5 Most Influential Pollutants

✅ Health Recommendation System

✅ Training Accuracy & Loss Dashboard

✅ Printable Prediction Report

✅ Railway Cloud Deployment

---

# 🧠 Machine Learning Pipeline

```
Raw Air Quality Dataset
           │
           ▼
Data Cleaning & Preprocessing
           │
           ▼
Feature Scaling (MinMaxScaler)
           │
           ▼
LSTM Neural Network Training
           │
           ▼
Model Prediction
           │
           ▼
SHAP Explainability
           │
           ▼
Health Recommendation
```

---

# 📊 Input Features

The model predicts AQI using the following atmospheric pollutants.

| Feature |
|----------|
| PM2.5 |
| PM10 |
| NO |
| NO₂ |
| NOx |
| NH₃ |
| CO |
| SO₂ |
| O₃ |
| Benzene |
| Toluene |
| Xylene |

---

# 📈 Model Architecture

• LSTM Layer

• Dropout Layer

• Dense Layer

• Dense Layer

• Softmax Output Layer

---

# ⚙️ Technologies Used

### Backend

- Python
- Flask
- TensorFlow
- Keras
- SHAP

### Machine Learning

- NumPy
- Pandas
- Scikit-learn
- Joblib

### Frontend

- HTML5
- CSS3
- JavaScript

### Deployment

- Railway
- GitHub

---

# 📂 Project Structure

```
AirPure-AI
│
├── app.py
├── train_lstm.py
├── shap_utils.py
├── requirements.txt
├── Start_AirPureAI.bat
│
├── templates
│   ├── home.html
│   ├── index.html
│   ├── results.html
│   ├── training_performance.html
│   └── error.html
│
├── static
│   ├── style.css
│   ├── training_plot.png
│   ├── shap_plot.png
│   └── images
│
├── airquality_lstm.h5
├── scaler_x.joblib
├── label_encoder.joblib
├── city_day_cleaned.csv
└── test.csv
```

---

# 🖥️ Installation

Clone the repository

```bash
git clone https://github.com/SHEMANTH25/AirPure-AI.git
```

Move into project folder

```bash
cd AirPure-AI
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python app.py
```

---

# 📈 Model Performance

✔ High Prediction Accuracy

✔ LSTM Deep Learning Architecture

✔ Explainable AI using SHAP

✔ Health Recommendation Engine

---

# 🌍 Deployment

The application is deployed using **Railway**.

Deployment Platform

- Railway

Version Control

- GitHub

---

# 🔮 Future Enhancements

- Real-time AQI API Integration
- Weather API Support
- PDF Report Generation
- Interactive SHAP Dashboard
- User Login System
- Historical Prediction Storage
- Mobile Application

---

# 👨‍💻 Developer

## S Hemanth Naidu

AI & Machine Learning Enthusiast

Python Developer

Deep Learning

Flask

TensorFlow

GitHub

https://github.com/SHEMANTH25

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is developed for educational, research and portfolio purposes.
