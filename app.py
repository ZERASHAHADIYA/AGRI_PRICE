from flask import Flask, render_template, request, jsonify, send_file, url_for
import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from datetime import datetime
import io

app = Flask(__name__)

# Load dataset & model (keep your current code here)...
DATA_PATH = r"C:\Users\HP\OneDrive\Desktop\aiml\crop_price_app\agri_price_dataset.xlsx"
MODEL_PATH = "price_predictor.pkl"
STATIC_IMG = "static/trend.png"
LAST_N = 30

df = pd.read_excel(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Commodity', 'Market', 'Date']).reset_index(drop=True)

group = df.groupby(['Commodity', 'Market'])
for lag in [1, 2, 3, 7]:
    df[f'lag_{lag}'] = group['Modal_Price'].shift(lag)
for win in [3, 7]:
    df[f'rollmean_{win}'] = group['Modal_Price'].shift(1).rolling(win).mean()

original_numeric = [
    'Min_Price', 'Max_Price', 'Arrivals_Tonnes', 'Rainfall_mm', 'Temperature_C',
    'Humidity_%', 'Soil_Moisture_%', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg',
    'Transport_Cost_Rs', 'Demand_Index'
]

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

crops = sorted(df['Commodity'].unique().tolist())
markets = sorted(df['Market'].unique().tolist())


def build_input_row(subset, model):
    if model and hasattr(model, "feature_names_in_"):
        feature_order = list(model.feature_names_in_)
    else:
        feature_order = original_numeric + [f"lag_{i}" for i in [1, 2, 3, 7]] + [f"rollmean_{w}" for w in [3, 7]]
    last = subset.iloc[-1]
    input_dict = {}
    for feat in feature_order:
        if feat in subset.columns:
            val = last[feat]
            if pd.isna(val):
                if feat.startswith("lag_"):
                    n = int(feat.split("_")[1])
                    if len(subset) > n:
                        val = subset['Modal_Price'].iloc[-1 - n]
                    else:
                        val = subset['Modal_Price'].mean()
                elif feat.startswith("rollmean_"):
                    w = int(feat.split("_")[1])
                    if len(subset) >= w:
                        val = subset['Modal_Price'].iloc[-w:].mean()
                    else:
                        val = subset['Modal_Price'].mean()
                else:
                    val = subset[feat].median() if feat in subset else subset['Modal_Price'].mean()
        else:
            val = subset['Modal_Price'].mean()
        if pd.isna(val):
            val = subset['Modal_Price'].mean()
        input_dict[feat] = val
    return pd.DataFrame([input_dict], columns=feature_order)
    
@app.route("/")
def index():
    return render_template("index.html", crops=crops, markets=markets)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    crop = data.get("crop")
    market = data.get("market")

    subset = df[(df['Commodity'] == crop) & (df['Market'] == market)].copy()
    subset = subset.sort_values("Date").reset_index(drop=True)

    if subset.empty:
        return jsonify({"error": "No historical data found for this Crop and Market."}), 404

    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    try:
        X_input = build_input_row(subset, model)
        pred_val = model.predict(X_input)[0]
        prediction = round(float(pred_val), 2)
        pred_date = datetime.today().strftime("%d-%m-%Y")
        return jsonify({"prediction": prediction, "pred_date": pred_date})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics", methods=["POST"])
def api_analytics():
    data = request.get_json()
    crop = data.get("crop")
    market = data.get("market")

    subset = df[(df['Commodity'] == crop) & (df['Market'] == market)].copy()
    subset = subset.sort_values("Date").reset_index(drop=True)

    if subset.empty:
        return jsonify({"error": "No historical data found for this Crop and Market."}), 404

    last_n = subset.tail(LAST_N)
    stats = {
        "Average Price": round(float(last_n["Modal_Price"].mean()), 2),
        "Min Price": round(float(last_n["Modal_Price"].min()), 2),
        "Max Price": round(float(last_n["Modal_Price"].max()), 2),
        "Total Arrivals": round(float(last_n["Arrivals_Tonnes"].sum()), 2),
        "Avg Rainfall (mm)": round(float(last_n["Rainfall_mm"].mean()), 2),
        "Avg Temperature (°C)": round(float(last_n["Temperature_C"].mean()), 2),
        "Avg Humidity (%)": round(float(last_n["Humidity_%"].mean()), 2),
        "Avg Soil Moisture (%)": round(float(last_n["Soil_Moisture_%"].mean()), 2),
    }

    # Generate chart
    plt.figure(figsize=(7, 3.5))
    plt.plot(last_n['Date'], last_n['Modal_Price'], marker='o', label=f'Historical (last {len(last_n)})')
    plt.title(f"Price Trend for {crop} in {market}")
    plt.xlabel("Date")
    plt.ylabel("Modal Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()
    img_bytes.seek(0)

    # Encode image in base64
    import base64
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode()

    return jsonify({"stats": stats, "chart_img": f"data:image/png;base64,{img_base64}"})


if __name__ == "__main__":
    app.run(debug=True)
