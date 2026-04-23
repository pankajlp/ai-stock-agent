
# models/lstm/predict.py

import numpy as np
import yfinance as yf
from keras.models import load_model
import joblib

def predict(symbol="RELIANCE.NS"):

    model = load_model(f"models/lstm/{symbol}_model.h5")
    scaler = joblib.load(f"models/lstm/{symbol}_scaler.pkl")

    df = yf.download(symbol, period="3mo")
    data = df[['Close']].values

    last_60 = data[-60:]
    scaled = scaler.transform(last_60)

    x_test = np.reshape(scaled, (1, 60, 1))

    pred = model.predict(x_test)
    pred_price = scaler.inverse_transform(pred)[0][0]

    last_price = data[-1][0]

    change_pct = ((pred_price - last_price) / last_price) * 100

    return {
        "predicted_price": float(pred_price),
        "last_price": float(last_price),
        "change_percent": float(change_pct)
    }
