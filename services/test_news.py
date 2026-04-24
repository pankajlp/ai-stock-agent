from services.news_service import get_news
from services.openai_service import analyze_news
from pipelines.sentiment_pipeline import aggregate_sentiment

news = get_news("Reliance")

raw = analyze_news(news)

final = aggregate_sentiment(raw)

print(final)