import numpy as np
import yfinance as yf
from keras.models import load_model
import joblib

def predict(symbol="RELIANCE.NS"):

    try:
        model = load_model(f"models/lstm/{symbol}_model.h5")
        scaler = joblib.load(f"models/lstm/{symbol}_scaler.pkl")

        df = yf.download(symbol, period="6mo")

        if df.empty:
            raise ValueError("No data fetched from yfinance")

        data = df[['Close']].dropna().values

        if len(data) < 60:
            df = yf.download(symbol, period="1y")
            data = df[['Close']].dropna().values

        if len(data) < 60:
            raise ValueError("Still not enough data after fallback")

        last_60 = data[-60:]

        scaled = scaler.transform(last_60)

        if np.isnan(scaled).any():
            raise ValueError("NaN found after scaling")

        x_test = np.reshape(scaled, (1, 60, 1))

        pred = model.predict(x_test,verbose=0)

        if np.isnan(pred).any():
            raise ValueError("Model returned NaN")

        pred_price = scaler.inverse_transform(pred)[0][0]
        last_price = data[-1][0]

        change_pct = ((pred_price - last_price) / last_price) * 100

        return {
            "predicted_price": float(pred_price),
            "last_price": float(last_price),
            "change_percent": float(change_pct)
        }

    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")

        return {
            "predicted_price": None,
            "last_price": None,
            "change_percent": 0
        }
def predict_from_array(model, scaler, data_window):
    """
    data_window: numpy array of shape (60, 1)
    """

    import numpy as np

    scaled = scaler.transform(data_window)

    x_test = np.reshape(scaled, (1, 60, 1))

    pred = model.predict(x_test, verbose=0)

    pred_price = scaler.inverse_transform(pred)[0][0]

    last_price = data_window[-1][0]

    change_pct = ((pred_price - last_price) / last_price) * 100

    return {
        "predicted_price": float(pred_price),
        "last_price": float(last_price),
        "change_percent": float(change_pct)
    }