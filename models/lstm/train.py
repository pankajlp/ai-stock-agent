
# models/lstm/train.py

import yfinance as yf
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime
from utils import prepare_data
import joblib

def train_model(symbol="RELIANCE.NS"):

    df = yf.download(symbol, start="2022-01-01", end=datetime.now())
    data = df[['Close']].values

    x_train, y_train, scaler = prepare_data(data)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    model.save(f"models/lstm/{symbol}_model.h5")
    joblib.dump(scaler, f"models/lstm/{symbol}_scaler.pkl")

    print("✅ Model trained and saved")

if __name__ == "__main__":
    train_model()
