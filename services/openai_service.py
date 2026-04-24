
from dotenv import load_dotenv
import os

load_dotenv()  # 🔥 THIS LINE IS CRITICAL

api_key = os.getenv("GROQ_API_KEY")

print("DEBUG API KEY:", api_key)  # 👈 TEMP DEBUG

from groq import Groq
client = Groq(api_key=api_key)
def analyze_news(news_list):

    combined_news = "\n".join(
        [f"{n['title']} - {n.get('description', '')}" for n in news_list]
    )

    prompt = f"""
    Analyze the following news and return JSON:

    1. Overall sentiment (positive, negative, neutral)
    2. Impact score (-1 to +1)
    3. Event type (geopolitical, policy, corporate, macro)
    4. Short reasoning

    News:
    {combined_news}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # 🔥 best free option
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content