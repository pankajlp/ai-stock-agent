from models.lstm.predict import predict
from services.news_service import get_news
from services.openai_service import analyze_news
from pipelines.sentiment_pipeline import aggregate_sentiment
from agents.decision_engine import make_decision
from agents.trading_agent import TradingAgent

symbol = "RELIANCE.NS"

agent = TradingAgent()

model_output = predict(symbol)

news = get_news("Reliance")
raw = analyze_news(news)
sentiment = aggregate_sentiment(raw)

decision = make_decision(model_output, sentiment, df)

price = model_output["last_price"]

print("Decision:", decision)

agent.act(decision, price)

print("Portfolio:", agent.status(price))