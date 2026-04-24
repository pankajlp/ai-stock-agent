import numpy as np
import pandas as pd

def moving_average(series, window=20):
    return series.rolling(window=window).mean()

def compute_rsi(series, period=14):
    delta = series.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def day_change_pct(series):
    if len(series) == 0:
        return 0.0
    return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100