from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# =======================
# Load dataset
# =======================
df = pd.read_excel(r"C:\Users\HP\OneDrive\Desktop\aiml\crop_price_app\agri_price_dataset.xlsx")

# Ensure correct types
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Commodity', 'Market', 'Date'])

# Dropdown values
crops = df["Commodity"].unique().tolist()
markets = df["Market"].unique().tolist()

# =======================
# Load model
# =======================
MODEL_PATH = "price_predictor.pkl"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print("❌ Error loading model:", e)
else:
    print("⚠️ Model file not found. Predictions won’t work until you save price_predictor.pkl")

# =======================
# Routes
# =======================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    stats = {}
    chart_path = None

    if request.method == "POST":
        crop = request.form["crop"]
        market = request.form["market"]

        # Subset data
        subset = df[(df["Commodity"] == crop) & (df["Market"] == market)].copy()

        if subset.empty:
            prediction = "No data available for this selection."
        else:
            # Simple features
            X_cols = [c for c in subset.columns if c not in ["Date", "Commodity", "Market", "Modal_Price"]]

            if model is not None:
                try:
                    latest = subset.iloc[-1:][X_cols]
                    prediction = model.predict(latest)[0]
                except Exception as e:
                    prediction = f"⚠️ Prediction error: {e}"
            else:
                prediction = "⚠️ Model not loaded."

            # Analytics
            stats = {
                "Average Price": round(subset["Modal_Price"].mean(), 2),
                "Min Price": round(subset["Modal_Price"].min(), 2),
                "Max Price": round(subset["Modal_Price"].max(), 2),
                "Total Arrivals": round(subset["Arrivals_Tonnes"].sum(), 2),
                "Avg Rainfall (mm)": round(subset["Rainfall_mm"].mean(), 2),
                "Avg Temperature (°C)": round(subset["Temperature_C"].mean(), 2),
                "Avg Humidity (%)": round(subset["Humidity_%"].mean(), 2),
                "Avg Soil Moisture (%)": round(subset["Soil_Moisture_%"].mean(), 2)
            }

            # Plot actual trend
            plt.figure(figsize=(8, 4))
            plt.plot(subset["Date"], subset["Modal_Price"], label="Actual Price", marker="o")
            plt.title(f"Price Trend for {crop} in {market}")
            plt.xlabel("Date")
            plt.ylabel("Modal Price")
            plt.legend()
            chart_path = os.path.join("static", "trend.png")
            plt.savefig(chart_path, bbox_inches="tight")
            plt.close()

    return render_template("index.html",
                           crops=crops,
                           markets=markets,
                           prediction=prediction,
                           stats=stats,
                           chart=chart_path)


if __name__ == "__main__":
    app.run(debug=True)
