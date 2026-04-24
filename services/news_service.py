

import requests

API_KEY = "8c20b0e9774e4b4aa644b5a11a4c0d90"

def get_news(query="RELIANCE"):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}&language=en"

    response = requests.get(url)
    data = response.json()

    articles = []

    for article in data.get("articles", [])[:5]:
        articles.append({
            "title": article["title"],
            "description": article["description"]
        })

    return articles