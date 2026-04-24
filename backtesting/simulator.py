
import yfinance as yf
import numpy as np
import joblib

from keras.models import load_model

from agents.trading_agent import TradingAgent
from agents.decision_engine import make_decision

from pipelines.sentiment_pipeline import aggregate_sentiment
from services.openai_service import analyze_news
from services.news_service import get_news

from models.lstm.predict import predict_from_array
def run_backtest(symbol="RELIANCE.NS"):

    df = yf.download(symbol, period="6mo")

    if df.empty or "Close" not in df.columns:
        raise ValueError("No valid data fetched")

    df = df.dropna().copy()

    if len(df) < 100:
        raise ValueError("Not enough data for backtest")

    # ✅ Load model
    model = load_model(f"models/lstm/{symbol}_model.h5")
    scaler = joblib.load(f"models/lstm/{symbol}_scaler.pkl")

    agent = TradingAgent()
    history = []

    # ✅ News (safe)
    try:
        news = get_news(symbol.split(".")[0])
        raw = analyze_news(news)
        sentiment = aggregate_sentiment(raw)
    except Exception as e:
        print("News error:", e)
        sentiment = {"final_score": 0, "sentiment": "neutral"}

    # 🔥 Rolling loop
    for i in range(60, len(df)):

        df_slice = df.iloc[:i]

        # ✅ Safety check
        if len(df_slice) < 60:
            continue

        window = df_slice[['Close']].values[-60:]

        current_price = float(df.iloc[i]['Close'])

        if np.isnan(current_price):
            continue

        # 🔥 Prediction
        try:
            model_output = predict_from_array(model, scaler, window)
        except Exception as e:
            print("Prediction error:", e)
            continue

        # 🔥 Decision (IMPORTANT FIX)
        decision = make_decision(model_output, sentiment, df_slice)

        # 🔥 Execute trade
        try:
            agent.act(decision, current_price)
        except Exception as e:
            print("Agent error:", e)
            continue

        portfolio = agent.status(current_price)

        history.append({
            "day": i,
            "price": round(current_price, 2),
            "action": decision["action"],
            "score": decision.get("score", 0),
            "portfolio": portfolio["portfolio_value"]
        })

    return history