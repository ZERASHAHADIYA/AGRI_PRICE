from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os

app = Flask(__name__)

DATA_PATH = r"C:\Users\HP\OneDrive\Desktop\aiml\crop_price_app\agri_price_dataset.xlsx"
MODEL_PATH = "price_predictor.pkl"
LAST_N = 30

# Load dataset
df = pd.read_excel(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Commodity','Market','Date']).reset_index(drop=True)

# Lag and rolling features
group = df.groupby(['Commodity','Market'])
for lag in [1,2,3,7]:
    df[f'lag_{lag}'] = group['Modal_Price'].shift(lag)
for win in [3,7]:
    df[f'rollmean_{win}'] = group['Modal_Price'].shift(1).rolling(win).mean()

original_numeric = [
    'Min_Price','Max_Price','Arrivals_Tonnes','Rainfall_mm','Temperature_C',
    'Humidity_%','Soil_Moisture_%','Fertilizer_Usage_kg','Pesticide_Usage_kg',
    'Transport_Cost_Rs','Demand_Index'
]

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
crops = sorted(df['Commodity'].unique().tolist())
markets = sorted(df['Market'].unique().tolist())

def build_input_row(subset):
    feature_order = list(model.feature_names_in_) if model and hasattr(model, "feature_names_in_") else \
        original_numeric + [f"lag_{i}" for i in [1,2,3,7]] + [f"rollmean_{w}" for w in [3,7]]
    last = subset.iloc[-1]
    row = {}
    for feat in feature_order:
        if feat in subset.columns:
            val = last.get(feat,np.nan)
            if pd.isna(val):
                if feat.startswith("lag_"):
                    n=int(feat.split("_")[1])
                    val=subset['Modal_Price'].iloc[-1-n] if len(subset)>n else subset['Modal_Price'].mean()
                elif feat.startswith("rollmean_"):
                    w=int(feat.split("_")[1])
                    val=subset['Modal_Price'].iloc[-w:].mean() if len(subset)>=w else subset['Modal_Price'].mean()
                else:
                    val=subset[feat].median() if feat in subset else subset['Modal_Price'].mean()
        else:
            val=subset['Modal_Price'].mean()
        if pd.isna(val): val=subset['Modal_Price'].mean()
        row[feat]=val
    return pd.DataFrame([row], columns=feature_order)

@app.route("/")
def index():
    return render_template("index.html", crops=crops, markets=markets)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data=request.get_json()
    crop=data.get("crop")
    market=data.get("market")
    subset=df[(df['Commodity']==crop) & (df['Market']==market)].copy().sort_values("Date").reset_index(drop=True)
    if subset.empty: return jsonify({"error":"No data available for this Crop & Market."})
    if model is None: return jsonify({"error":"Prediction model not loaded."})
    try:
        X_input = build_input_row(subset)
        pred_val = round(float(model.predict(X_input)[0]),2)
        pred_date = datetime.today().strftime("%d-%m-%Y")
        return jsonify({"prediction":pred_val, "pred_date":pred_date})
    except Exception as e:
        return jsonify({"error":str(e)})

@app.route("/api/analytics", methods=["POST"])
def api_analytics():
    data=request.get_json()
    crop=data.get("crop")
    market=data.get("market")
    subset=df[(df['Commodity']==crop) & (df['Market']==market)].copy().sort_values("Date").reset_index(drop=True)
    if subset.empty:
        return jsonify({"error":"No data available for this Crop & Market.","stats":{},"chart_img":""})
    last_n=subset.tail(LAST_N)
    stats={
        "Average Price": round(float(last_n["Modal_Price"].mean()),2),
        "Min Price": round(float(last_n["Modal_Price"].min()),2),
        "Max Price": round(float(last_n["Modal_Price"].max()),2),
        "Total Arrivals": round(float(last_n["Arrivals_Tonnes"].sum()),2),
        "Avg Rainfall (mm)": round(float(last_n["Rainfall_mm"].mean()),2),
        "Avg Temperature (°C)": round(float(last_n["Temperature_C"].mean()),2),
        "Avg Humidity (%)": round(float(last_n["Humidity_%"].mean()),2),
        "Avg Soil Moisture (%)": round(float(last_n["Soil_Moisture_%"].mean()),2)
    }
    plt.figure(figsize=(7,3.5))
    plt.plot(last_n['Date'], last_n['Modal_Price'], marker='o', color="#2e7d32", label='Price Trend')
    plt.title(f"{crop} Price Trend in {market}")
    plt.xlabel("Date")
    plt.ylabel("Modal Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    img_bytes=io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
    return jsonify({"stats":stats, "chart_img":f"data:image/png;base64,{img_base64}"})


from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import base64

@app.route("/api/download", methods=["POST"])
def api_download():
    data = request.get_json()
    crop = data.get("crop")
    market = data.get("market")
    prediction = data.get("prediction")
    stats = data.get("stats")
    chart_base64 = data.get("chart_img").split(",")[1] if data.get("chart_img") else None

    # Create PDF in memory
    pdf_bytes = io.BytesIO()
    c = canvas.Canvas(pdf_bytes, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, "Agri Price Prediction Report")
    y -= 40

    # Crop & Market
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Crop: {crop}")
    y -= 20
    c.drawString(margin, y, f"Market: {market}")
    y -= 30

    # Predicted Price
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, f"Predicted Price: ₹{prediction}")
    y -= 30

    # Stats
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Analytics Stats:")
    y -= 20
    c.setFont("Helvetica", 12)
    for key, val in stats.items():
        c.drawString(margin+20, y, f"{key}: {val}")
        y -= 18

    # Chart
    if chart_base64:
        chart_data = base64.b64decode(chart_base64)
        chart_img = Image.open(io.BytesIO(chart_data))
        chart_reader = ImageReader(chart_img)
        c.drawImage(chart_reader, margin, y - 250, width=width - 2*margin, height=250)
    
    c.showPage()
    c.save()
    pdf_bytes.seek(0)
    
    return send_file(pdf_bytes, as_attachment=True, download_name="price_report.pdf", mimetype="application/pdf")

if __name__=="__main__":
    app.run(debug=True)
