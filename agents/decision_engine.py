import numpy as np


def make_decision(model_output, sentiment, df):

    # -------------------------
    # 🔒 SAFETY CHECKS
    # -------------------------
    if df is None or df.empty or "Close" not in df.columns:
        return {"action": "HOLD", "reason": "No data", "score": 0}

    close = df["Close"].dropna()

    if len(close) < 50:
        return {"action": "HOLD", "reason": "Not enough data", "score": 0}

    # -------------------------
    # 🧠 BASE SIGNAL
    # -------------------------
    try:
        model_score = float(model_output.get("change_percent", 0)) / 100
    except:
        model_score = 0

    try:
        news_score = float(sentiment.get("final_score", 0))
    except:
        news_score = 0

    final_score = model_score + news_score

    action = "HOLD"
    if abs(final_score) > 0.01:
        action = "BUY" if final_score > 0 else "SELL"

    # -------------------------
    # 🔥 FORCE SCALAR VALUES
    # -------------------------
    # current price
    current_price_raw = close.iloc[-1]
    if hasattr(current_price_raw, "values"):
        current_price = float(current_price_raw.values[0])
    else:
        current_price = float(current_price_raw)

    # -------------------------
    # 🔥 TREND FILTER (MA20)
    # -------------------------
    ma20_series = close.rolling(20).mean()
    ma20_raw = ma20_series.iloc[-1]

    if hasattr(ma20_raw, "values"):
        ma20 = float(ma20_raw.values[0])
    else:
        ma20 = float(ma20_raw)

    if not np.isnan(ma20):
        if action == "BUY" and current_price < ma20:
            return {
                "action": "HOLD",
                "reason": "Downtrend",
                "score": round(final_score, 4)
            }

    # -------------------------
    # 🔥 CRASH FILTER
    # -------------------------
    first_price_raw = close.iloc[0]

    if hasattr(first_price_raw, "values"):
        first_price = float(first_price_raw.values[0])
    else:
        first_price = float(first_price_raw)

    day_change = ((current_price - first_price) / first_price) * 100

    if not np.isnan(day_change):
        if action == "BUY" and day_change < -2:
            return {
                "action": "HOLD",
                "reason": "Market falling",
                "score": round(final_score, 4)
            }

    # -------------------------
    # 🔥 RSI FILTER
    # -------------------------
    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))

    rsi_raw = rsi.iloc[-1]

    if hasattr(rsi_raw, "values"):
        rsi_val = float(rsi_raw.values[0])
    else:
        rsi_val = float(rsi_raw)

    if not np.isnan(rsi_val):
        if action == "BUY" and rsi_val < 25:
            return {
                "action": "HOLD",
                "reason": "Extreme panic",
                "score": round(final_score, 4)
            }

        if action == "SELL" and rsi_val > 75:
            return {
                "action": "HOLD",
                "reason": "Overbought",
                "score": round(final_score, 4)
            }

    # -------------------------
    # ✅ FINAL OUTPUT
    # -------------------------
    return {
        "action": action,
        "score": round(final_score, 4)
    }