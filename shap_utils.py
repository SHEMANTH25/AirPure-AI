import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
FEATURE_NAMES = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene"
]
def predict_function(model):

    def predict(x):

        x = np.array(x)

        x = x.reshape(
            x.shape[0],
            1,
            x.shape[1]
        )

        return model.predict(
            x,
            verbose=0
        )

    return predict
def create_background():

    df = pd.read_csv("city_day_cleaned.csv")

    background = df[FEATURE_NAMES].sample(
        n=30,
        random_state=42
    )

    return background
def create_explainer(model, scaler_x):

    # Load background data
    background = scaler_x.transform(
        create_background()
    )

    # Reshape for LSTM
    background = background.reshape(
        background.shape[0],
        1,
        background.shape[1]
    )

    # Create SHAP DeepExplainer
    explainer = shap.DeepExplainer(
        model,
        background
    )
    return explainer
def generate_shap_plot(

    model,

    sample_scaled,

    scaler_x

):
    try:
        if not os.path.exists("static"):
            os.makedirs("static")

        explainer = create_explainer(
            model,
            scaler_x
        )

        # Reshape input for LSTM
        sample = sample_scaled.reshape(
            sample_scaled.shape[0],
            1,
            sample_scaled.shape[1]
        )

        # Generate SHAP values using DeepExplainer
        shap_values = explainer.shap_values(sample)

        # Handle DeepExplainer output
        if isinstance(shap_values, list):
            values = shap_values[0]
        else:
            values = shap_values

        # Remove batch dimension if present
        if values.ndim == 3:
            values = values[0]

        # ==========================================
        # Calculate SHAP Importance
        # ==========================================

        # Calculate feature importance
        if values.ndim == 1:
            importance = np.abs(values)
        else:
            importance = np.mean(np.abs(values), axis=0)

        sorted_index = np.argsort(importance)[::-1]
        feature_names = np.array(FEATURE_NAMES)[sorted_index]
        importance = importance[sorted_index]

        # ==========================================
        # Bar Colors
        # ==========================================

        colors = [
            "#ff1744",  # Red
            "#ff9100",  # Orange
            "#ffd600",  # Yellow
            "#00c853",  # Green
            "#00b8d4",  # Cyan
            "#2979ff",  # Blue
            "#651fff",  # Purple
            "#d500f9",  # Pink
            "#795548",  # Brown
            "#607d8b",  # Blue Grey
            "#9e9e9e",  # Grey
            "#cfd8dc"   # Light Grey
        ]

        # ==========================================
        # Draw Professional Chart
        # ==========================================

        plt.figure(figsize=(9,5))
        bars = plt.barh(
            feature_names,
            importance,
            color=colors[:len(feature_names)],
            edgecolor="black",
            linewidth=1
        )
        plt.gca().invert_yaxis()

        # ==========================================
        # Titles
        # ==========================================

        plt.title(
            "SHAP Feature Importance",
            fontsize=20,
            fontweight="bold",
            pad=20
        )
        plt.xlabel(
            "SHAP Value",
            fontsize=14,
            fontweight="bold"
        )
        plt.ylabel(
            "Air Pollutants",
            fontsize=14,
            fontweight="bold"
        )

        plt.grid(
            axis="x",
            linestyle="--",
            alpha=0.35
        )

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va="center",
                fontsize=10,
                fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(
            "static/shap_plot.png",
            dpi=250,
            bbox_inches="tight",
            facecolor="white"
        )
        plt.close()

        top_features = []
        for rank, i in enumerate(sorted_index[:5]):
            if values.ndim == 1:
                score = float(np.abs(values[i]))
            else:
                score = float(np.mean(np.abs(values[:, i])))

            if rank == 0:
                impact = "Very High"
            elif rank == 1:
                impact = "High"
            elif rank == 2:
                impact = "Medium"
            elif rank == 3:
                impact = "Low"
            else:
                impact = "Very Low"

            top_features.append({
                "name": FEATURE_NAMES[i],
                "value": round(score, 4),
                "impact": impact
            })

        return values, top_features
    except Exception as e:
        print("SHAP Error:", e)
        return None, [
            {"name":"PM2.5","value":0,"impact":"N/A"},
            {"name":"PM10","value":0,"impact":"N/A"},
            {"name":"NO2","value":0,"impact":"N/A"}
        ]

def build_results(

    model,

    sample_reshaped,

    sample_scaled,

    scaler_x,

    label_encoder,

    get_health_advice,

    render_template

):

    # ---------------------------------
    # Generate SHAP
    # ---------------------------------

    shap_values, top_features = generate_shap_plot(

        model,

        sample_scaled,

        scaler_x

    )

    # ---------------------------------
    # Predict AQI
    # ---------------------------------

    prediction = model.predict(

        sample_reshaped,

        verbose=0

    )

    predicted_class = np.argmax(

        prediction,

        axis=1

    )[0]

    result = label_encoder.inverse_transform(

        [predicted_class]

    )[0]

    confidence = round(

        float(np.max(prediction) * 100),

        2

    )

    conclusion = get_health_advice(

        result

    )

    # ---------------------------------
    # AI Explanation
    # ---------------------------------

    explanation = (

        f"The LSTM model predicted "

        f"'{result}' "

        f"with {confidence}% confidence. "

        f"The most influential pollutant "

        f"was {top_features[0]['name']} "

        f"(SHAP Value: {top_features[0]['value']}). "

        f"{top_features[1]['name']} "

        f"and "

        f"{top_features[2]['name']} "

        f"also contributed significantly "

        f"to the prediction."

    )

    return render_template(

        "results.html",

        result=result,

        confidence=confidence,

        conclusion=conclusion,

        shap_image="shap_plot.png",

        top_features=top_features,

        explanation=explanation

    )




