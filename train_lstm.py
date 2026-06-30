import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

def train_lstm_classification(data_file):
    print("Loading cleaned data...")
    df = pd.read_csv(data_file)
    
    # All pollutant features for maximum accuracy
    features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    target = 'AQI_Bucket'
    
    X = df[features].values
    y_raw = df[target].values
    
    # Encoding Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    y_categorical = to_categorical(y_encoded)
    
    # Scaling Features
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    
    # Reshape for LSTM: [samples, time_steps, features]
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Powerful Model Architecture to hit 98%+ Accuracy
    model = Sequential([
        LSTM(512, activation='relu', input_shape=(1, len(features)), return_sequences=True),
        BatchNormalization(),
        LSTM(256, activation='relu', return_sequences=True),
        BatchNormalization(),
        LSTM(128, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training powerful model with 12 features...")
    early_stop = EarlyStopping(monitor='accuracy', patience=20, restore_best_weights=True)
    
    history = model.fit(X_lstm, y_categorical, epochs=200, batch_size=64, verbose=1, callbacks=[early_stop])
    
    print("Generating training plots...")
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy', color='#6366f1', linewidth=3)
    plt.title('Training Accuracy', fontsize=14, pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss', color='#ef4444', linewidth=3)
    plt.title('Training Loss', fontsize=14, pad=15)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('static/training_plot.png', dpi=300, transparent=True)
    plt.close()
    
    print("Saving model and artifacts...")
    model.save('airquality_lstm.h5')
    joblib.dump(scaler_x, 'scaler_x.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    
    final_acc = max(history.history['accuracy'])
    print(f"Training complete! Peak Accuracy: {final_acc*100:.2f}%")

if __name__ == "__main__":
    train_lstm_classification('city_day_cleaned.csv')
