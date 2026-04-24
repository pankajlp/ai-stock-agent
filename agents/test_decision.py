from models.lstm.predict import predict
from services.news_service import get_news
from services.openai_service import analyze_news
from pipelines.sentiment_pipeline import aggregate_sentiment
from agents.decision_engine import make_decision

symbol = "RELIANCE.NS"

model_output = predict(symbol)

news = get_news("Reliance")
raw = analyze_news(news)
sentiment = aggregate_sentiment(raw)

decision = make_decision(model_output, sentiment,df)

print("MODEL:", model_output)
print("NEWS:", sentiment)
print("FINAL DECISION:", decision)