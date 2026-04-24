import streamlit as st
import yfinance as yf
import joblib
import numpy as np
import os
try:
    from keras.models import load_model
except:
    load_model = None
import plotly.graph_objects as go
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
# Fix imports
#import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lstm.predict import predict_from_array
from services.news_service import get_news
from services.openai_service import analyze_news
from pipelines.sentiment_pipeline import aggregate_sentiment
from agents.decision_engine import make_decision
from agents.trading_agent import TradingAgent

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="AI Stock Agent", layout="wide")
st.title("📈 AI Stock Trading Dashboard")

# -------------------------------
# Select Stock
# -------------------------------
symbol = st.selectbox(
    "Select Stock",
    ["RELIANCE.NS", "AAPL", "TCS.NS", "INFY.NS"]
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_components(symbol):
    model_path = f"models/lstm/{symbol}_model.h5"
    scaler_path = f"models/lstm/{symbol}_scaler.pkl"

    if not os.path.exists(model_path):
        return None, None

    if load_model is None:
        model = None
    else:
        model = load_model(f"models/lstm/{symbol}_model.h5")
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_components(symbol)

if model is None:
    st.error(f"❌ Model not found for {symbol}")
    st.stop()

# -------------------------------
# Agent + Trade Log
# -------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = TradingAgent()

if "trade_log" not in st.session_state:
    st.session_state.trade_log = []

agent = st.session_state.agent
trade_log = st.session_state.trade_log

# -------------------------------
# Data
# -------------------------------
@st.cache_data(ttl=30)
def get_data(symbol):
    return yf.download(symbol, period="5d", interval="5m", auto_adjust=False)

df = get_data(symbol)

if df.empty:
    st.error("No data available")
    st.stop()

df = df.dropna()

# -------------------------------
# Prediction
# -------------------------------
data = df[['Close']].values

if len(data) < 60:
    st.warning("Not enough data")
    st.stop()

window = data[-60:]
current_price = float(data[-1][0])

if model is None:
    model_output = {
        "predicted_price": current_price,
        "change_percent": 0
    }
else:
    model_output = predict_from_array(model, scaler, window)

# -------------------------------
# News
# -------------------------------
@st.cache_data(ttl=600)
def get_cached_news(symbol):
    try:
        news = get_news(symbol)
        raw = analyze_news(news)
        sentiment = aggregate_sentiment(raw)
        return news, sentiment
    except:
        return [], {"final_score": 0, "sentiment": "neutral"}

news, sentiment = get_cached_news(symbol.split(".")[0])

# -------------------------------
# Decision
# -------------------------------
decision = make_decision(model_output, sentiment, df)

# -------------------------------
# Trade
# -------------------------------
if st.button("Run Trade"):

    agent.act(decision, current_price)

    trade_log.append({
        "price": current_price,
        "action": decision["action"],
        "time": df.index[-1]
    })

# -------------------------------
# Metrics
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("💰 Price", f"{current_price:.2f}")
col2.metric("📊 Predicted %", f"{model_output['change_percent']:.2f}%")
col3.metric("⚡ Decision", decision["action"])

if "reason" in decision:
    st.warning(decision["reason"])

# -------------------------------
# Portfolio
# -------------------------------
st.subheader("💼 Portfolio")
portfolio = agent.status(current_price)
st.write(portfolio)

# -------------------------------
# 📊 CHART WITH MARKERS
# -------------------------------
st.subheader("📊 Advanced Chart")

chart_df = df.copy()

chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
chart_df["MA50"] = chart_df["Close"].rolling(50).mean()

fig = go.Figure()

# Candles
fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df['Open'],
    high=chart_df['High'],
    low=chart_df['Low'],
    close=chart_df['Close'],
    name='Price'
))

# MAs
fig.add_trace(go.Scatter(
    x=chart_df.index,
    y=chart_df['MA20'],
    name="MA20"
))

fig.add_trace(go.Scatter(
    x=chart_df.index,
    y=chart_df['MA50'],
    name="MA50"
))

# 🔥 TRADE MARKERS
for trade in trade_log:

    color = "green" if trade["action"] == "BUY" else "red"

    fig.add_trace(go.Scatter(
        x=[trade["time"]],
        y=[trade["price"]],
        mode="markers",
        marker=dict(color=color, size=10),
        name=trade["action"]
    ))

fig.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# RSI
# -------------------------------
delta = chart_df['Close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / (loss + 1e-9)
chart_df['RSI'] = 100 - (100 / (1 + rs))

st.subheader("📉 RSI")
st.line_chart(chart_df["RSI"])

# -------------------------------
# PERFORMANCE
# -------------------------------
st.subheader("📈 Performance")

portfolio_values = [t["price"] for t in trade_log]

if len(portfolio_values) > 1:
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

    st.write({
        "Return %": round(total_return * 100, 2),
        "Trades": len(trade_log)
    })

# -------------------------------
# Multi Stock Signals
# -------------------------------
st.subheader("📊 Multi Stock Signals")

symbols = ["RELIANCE.NS", "AAPL", "TCS.NS", "INFY.NS"]

results = []

for s in symbols:

    m, sc = load_components(s)
    if m is None:
        continue

    df_multi = get_data(s)
    if df_multi.empty:
        continue

    df_multi = df_multi.dropna()

    if len(df_multi) < 60:
        continue

    window_multi = df_multi[['Close']].values[-60:]
    price_multi = float(df_multi['Close'].values[-1])

    model_out = predict_from_array(m, sc, window_multi)

    news_m, sentiment_m = get_cached_news(s.split(".")[0])

    decision_m = make_decision(model_out, sentiment_m, df_multi)

    results.append({
        "symbol": s,
        "price": price_multi,
        "decision": decision_m["action"],
        "score": decision_m.get("score", 0)
    })

if results:
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    for r in results:
        st.write(f"### {r['symbol']}")
        st.write(f"Price: {r['price']:.2f}")
        st.write(f"Decision: {r['decision']}")
        st.write(f"Score: {r['score']}")
        st.write("---")

    best = results[0]
    st.success(f"🔥 Best Opportunity: {best['symbol']} ({best['decision']})")