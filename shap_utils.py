import os
import shap
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# Feature Names
# ============================================================

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


# ============================================================
# Background Dataset
# ============================================================

def create_background():

    df = pd.read_csv("city_day_cleaned.csv")

    background = df[FEATURE_NAMES].sample(

        n=20,

        random_state=5

    )

    return background


# ============================================================
# Create Gradient Explainer
# ============================================================

def create_explainer(

    model,

    scaler_x

):

    background = scaler_x.transform(

        create_background()

    )

    background = background.reshape(

        background.shape[0],

        1,

        background.shape[1]

    )

    explainer = shap.GradientExplainer(

        model,

        background

    )

    return explainer


# ============================================================
# Generate SHAP Plot
# ============================================================

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

        sample = sample_scaled.reshape(

            sample_scaled.shape[0],

            1,

            sample_scaled.shape[1]

        )

        explanation = explainer(sample)

        if hasattr(explanation, "values"):

            values = explanation.values

        else:

            values = explanation

        values = np.squeeze(np.array(values))

        if values.ndim == 1:

            importance = np.abs(values)

        else:

            importance = np.mean(

                np.abs(values),

                axis=0

            )

        importance = np.squeeze(importance)

        sorted_index = np.argsort(

            importance

        )[::-1]

        feature_names = np.array(

            FEATURE_NAMES

        )[sorted_index]

        importance = importance[

            sorted_index

        ]

        colors = [
            "#ff1744",
            "#ff9100",
            "#ffd600",
            "#00c853",
            "#00b8d4",
            "#2979ff",
            "#651fff",
            "#d500f9",
            "#795548",
            "#607d8b",
            "#9e9e9e",
            "#cfd8dc"
        ]

        plt.figure(figsize=(11, 6))

        bars = plt.barh(

            feature_names,

            importance,

            color=colors[:len(feature_names)],

            edgecolor="black",

            linewidth=1.2

        )

        plt.gca().invert_yaxis()

        plt.title(

            "SHAP Feature Importance",

            fontsize=20,

            fontweight="bold",

            pad=20

        )

        plt.xlabel(

            "Importance Score",

            fontsize=14,

            fontweight="bold"

        )

        plt.ylabel(

            "Air Pollutants",

            fontsize=14,

            fontweight="bold"

        )

        plt.xticks(fontsize=11)

        plt.yticks(fontsize=11)

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

            dpi=300,

            bbox_inches="tight",

            facecolor="white"

        )

        plt.close()

        top_features = []

        for rank, index in enumerate(sorted_index[:5]):

            score = float(importance[rank])

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

            top_features.append(

                {

                    "name": FEATURE_NAMES[index],

                    "value": round(score, 4),

                    "impact": impact

                }

            )

        return values, top_features

    except Exception as e:

        print("SHAP Error:", e)

        return None, [

            {

                "name": "PM2.5",

                "value": 0,

                "impact": "N/A"

            },

            {

                "name": "PM10",

                "value": 0,

                "impact": "N/A"

            },

            {

                "name": "NO2",

                "value": 0,

                "impact": "N/A"

            }

        ]
    # ============================================================
# Build Results Page
# ============================================================

def build_results(

    model,

    sample_reshaped,

    sample_scaled,

    scaler_x,

    label_encoder,

    get_health_advice,

    render_template

):

    # ---------------------------------------------
    # Generate SHAP Explanation
    # ---------------------------------------------

    shap_values, top_features = generate_shap_plot(

        model,

        sample_scaled,

        scaler_x

    )

    # ---------------------------------------------
    # Model Prediction
    # ---------------------------------------------

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

    # ---------------------------------------------
    # AI Explanation
    # ---------------------------------------------

    if top_features and len(top_features) >= 3:

        explanation = (

            f"The LSTM model predicted "

            f"'{result}' "

            f"with {confidence}% confidence. "

            f"The most influential pollutant was "

            f"{top_features[0]['name']} "

            f"(Importance Score: {top_features[0]['value']}). "

            f"{top_features[1]['name']} "

            f"and "

            f"{top_features[2]['name']} "

            f"also played a significant role in "

            f"the prediction."

        )

    else:

        explanation = (

            f"The LSTM model predicted "

            f"'{result}' "

            f"with {confidence}% confidence."

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

