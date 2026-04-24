import time
import yfinance as yf
import joblib
import numpy as np
from keras.models import load_model

from agents.trading_agent import TradingAgent
from agents.decision_engine import make_decision
from pipelines.sentiment_pipeline import aggregate_sentiment
from services.openai_service import analyze_news
from services.news_service import get_news
from models.lstm.predict import predict_from_array


def run_live(symbol="RELIANCE.NS"):

    print("🚀 Starting live trading agent...\n")

    model = load_model(f"models/lstm/{symbol}_model.h5")
    scaler = joblib.load(f"models/lstm/{symbol}_scaler.pkl")

    agent = TradingAgent()

    while True:

        try:
            df = yf.download(symbol, period="5d", interval="5m")

            if df.empty:
                print("⚠️ No data")
                time.sleep(60)
                continue

            data = df[['Close']].dropna().values

            if len(data) < 60:
                print("⏳ Not enough data yet")
                time.sleep(60)
                continue

            window = data[-60:]
            current_price = float(data[-1][0])

            # 🧠 prediction
            model_output = predict_from_array(model, scaler, window)

            # 🗞️ news (can cache later)
            news = get_news("Reliance")
            raw = analyze_news(news)
            sentiment = aggregate_sentiment(raw)

            decision = make_decision(model_output, sentiment)

            print("\n📊 PRICE:", current_price)
            print("🧠 MODEL:", model_output)
            print("📰 NEWS:", sentiment)
            print("⚡ DECISION:", decision)

            agent.act(decision, current_price)

            print("💰 PORTFOLIO:", agent.status(current_price))

        except Exception as e:
            print("❌ Error:", e)

        # ⏱️ wait 5 minutes
        time.sleep(300)