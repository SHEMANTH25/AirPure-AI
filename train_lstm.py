import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def train_lstm_classification(data_file):

    print("Loading cleaned data...")

    df = pd.read_csv(data_file)

    # 12 pollutant features
    features = [
        'PM2.5',
        'PM10',
        'NO',
        'NO2',
        'NOx',
        'NH3',
        'CO',
        'SO2',
        'O3',
        'Benzene',
        'Toluene',
        'Xylene'
    ]

    target = "AQI_Bucket"

    X = df[features].values
    y_raw = df[target].values

    # -----------------------------
    # Encode Labels
    # -----------------------------

    le = LabelEncoder()

    y_encoded = le.fit_transform(y_raw)

    num_classes = len(le.classes_)

    y_categorical = to_categorical(y_encoded)

    # -----------------------------
    # Scale Features
    # -----------------------------

    scaler_x = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(X)

    # -----------------------------
    # Reshape for LSTM
    # -----------------------------

    X_lstm = X_scaled.reshape(
        X_scaled.shape[0],
        1,
        X_scaled.shape[1]
    )

    # =====================================================
    # Optimized Lightweight LSTM Model
    # =====================================================

    model = Sequential([

        LSTM(
            64,
            input_shape=(1, len(features)),
            return_sequences=False
        ),

        Dropout(0.20),

        Dense(
            32,
            activation="relu"
        ),

        Dropout(0.20),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            num_classes,
            activation="softmax"
        )

    ])

    model.compile(

        optimizer=Adam(
            learning_rate=0.001
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    print("Training Optimized LSTM...")

    early_stop = EarlyStopping(

        monitor="loss",

        patience=10,

        restore_best_weights=True

    )

    history = model.fit(

        X_lstm,

        y_categorical,

        epochs=100,

        batch_size=32,

        verbose=1,

        callbacks=[early_stop]

    )

    # -----------------------------
    # Training Graph
    # -----------------------------

    plt.style.use("dark_background")

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)

    plt.plot(
        history.history["accuracy"],
        color="#00e676",
        linewidth=3,
        label="Accuracy"
    )

    plt.title("Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.subplot(1,2,2)

    plt.plot(
        history.history["loss"],
        color="#ff1744",
        linewidth=3,
        label="Loss"
    )

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "static/training_plot.png",
        dpi=300,
        transparent=True
    )

    plt.close()

    # -----------------------------
    # Save Model
    # -----------------------------

    print("Saving Model...")

    model.save("airquality_lstm.h5")

    joblib.dump(
        scaler_x,
        "scaler_x.joblib"
    )

    joblib.dump(
        le,
        "label_encoder.joblib"
    )

    final_acc = max(history.history["accuracy"])

    print(
        f"\nTraining Completed Successfully!\nPeak Accuracy : {final_acc*100:.2f}%"
    )


if __name__ == "__main__":

    train_lstm_classification(
        "city_day_cleaned.csv"
    )