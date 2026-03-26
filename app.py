# =========================
# Re-MiMa WEATHER - FINAL DEPLOYABLE VERSION
# =========================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

# =========================
# FIXED PATH (IMPORTANT)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# =========================
# UI CITIES
# =========================
city_display = {
    "chennai": "Chennai",
    "pune": "Pune",
    "ahmedabad": "Ahmedabad",
    "jaipur": "Jaipur",
    "lucknow": "Lucknow"
}

# =========================
# NEAREST MODEL CITY
# =========================
city_to_model = {
    "chennai": "bangalore",
    "pune": "mumbai",
    "ahmedabad": "mumbai",
    "jaipur": "delhi",
    "lucknow": "delhi"
}

# =========================
# LOAD MODELS
# =========================
models = {}
scalers = {}
features_map = {}

model_cities = ["bangalore", "delhi", "hyderabad", "kolkata", "mumbai"]

for city in model_cities:
    try:
        models[f"{city}_micro"] = joblib.load(os.path.join(MODEL_DIR, f"{city}_micro_model.pkl"))
        models[f"{city}_macro"] = joblib.load(os.path.join(MODEL_DIR, f"{city}_macro_model.pkl"))

        scalers[f"{city}_micro"] = joblib.load(os.path.join(MODEL_DIR, f"{city}_scaler.pkl"))
        scalers[f"{city}_macro"] = joblib.load(os.path.join(MODEL_DIR, f"{city}_macro_scaler.pkl"))

        features_map[f"{city}_micro"] = joblib.load(os.path.join(MODEL_DIR, f"{city}_features.pkl"))
        features_map[f"{city}_macro"] = joblib.load(os.path.join(MODEL_DIR, f"{city}_macro_features.pkl"))

        print(f"✅ Loaded {city}")

    except Exception as e:
        print(f"❌ {city} error: {e}")

print("🚀 Re-MiMa READY")

# =========================
# PREDICTION FUNCTION
# =========================
def re_mima_predict(place, month, hour):
    try:
        model_city = city_to_model.get(place)
        if model_city is None:
            return None

        micro_features = features_map[f"{model_city}_micro"]
        macro_features = features_map[f"{model_city}_macro"]

        # create input
        micro_input = pd.DataFrame([{f: 0 for f in micro_features}])
        macro_input = pd.DataFrame([{f: 0 for f in macro_features}])

        # set features
        for df in [micro_input, macro_input]:
            if "Month" in df.columns:
                df["Month"] = month
            if "month" in df.columns:
                df["month"] = month
            if "hour" in df.columns:
                df["hour"] = hour

        # predict micro
        X_micro = scalers[f"{model_city}_micro"].transform(micro_input)
        micro_pred = models[f"{model_city}_micro"].predict(X_micro)[0]

        # predict macro
        X_macro = scalers[f"{model_city}_macro"].transform(macro_input)
        macro_pred = models[f"{model_city}_macro"].predict(X_macro)[0]

        # adjustment
        adjust = {
            "chennai": 1.2,
            "pune": -0.8,
            "ahmedabad": 1.9,
            "jaipur": 1.6,
            "lucknow": 0.4
        }

        return {
            "temperature": float(macro_pred[0]) + adjust.get(place, 0),
            "humidity": float(micro_pred[0]),
            "rainfall": float(micro_pred[1]),
            "wind": float(micro_pred[2]),
        }

    except Exception as e:
        print("❌ ERROR:", e)
        return None

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    error = None

    if request.method == "POST":
        try:
            place = request.form.get("place", "").strip().lower()

            if not place:
                return render_template("index.html", error="⚠️ Please select a city", city_display=city_display)

            time_str = request.form.get("time")

            if not time_str:
                return render_template("index.html", error="⚠️ Please select date & time", city_display=city_display)

            dt = datetime.fromisoformat(time_str)

            if dt.year < 2021 or dt.year > 2024:
                return render_template("index.html", error="⚠️ Only 2021–2024 supported", city_display=city_display)

            month = dt.month
            hour = dt.hour

            results = []
            last_temp = None

            for i in range(1, 6):
                future_hour = (hour + i) % 24
                pred = re_mima_predict(place, month, future_hour)

                if pred:
                    temp = pred["temperature"] + (i * 0.4)

                    if last_temp:
                        temp = (temp + last_temp) / 2

                    humidity = pred["humidity"] - (i * 0.6)

                    last_temp = temp

                    temp = round(temp, 1)
                    humidity = round(humidity, 1)

                    condition = "Hot ☀️" if temp > 35 else "Cold ❄️" if temp < 20 else "Moderate 🌤️"

                    results.append({
                        "hour": future_hour,
                        "temperature": temp,
                        "humidity": humidity,
                        "rainfall": round(pred["rainfall"], 1),
                        "wind": round(pred["wind"], 1),
                        "condition": condition
                    })

        except Exception as e:
            error = str(e)

    return render_template("index.html", results=results, error=error, city_display=city_display)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)