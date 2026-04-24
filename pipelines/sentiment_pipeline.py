import json
import re

def aggregate_sentiment(llm_output):

    if not llm_output:
        return {
            "final_score": 0,
            "sentiment": "neutral"
        }

    try:
        # 🔍 Extract JSON using regex
        match = re.search(r"\[.*\]", llm_output, re.DOTALL)

        if not match:
            raise ValueError("No JSON found")

        json_str = match.group(0)

        data = json.loads(json_str)

    except Exception:
        # 🔥 Fallback (VERY IMPORTANT)
        return {
            "final_score": 0,
            "sentiment": "neutral"
        }

    # weights
    weights = {
        "geopolitical": 1.5,
        "macro": 1.4,
        "policy": 1.2,
        "corporate": 1.0
    }

    weighted_sum = 0
    weight_total = 0

    for item in data:
        try:
            score = float(item["impact_score"])
            event = item["event_type"]

            weight = weights.get(event, 1.0)

            weighted_sum += score * weight
            weight_total += weight
        except:
            continue

    if weight_total == 0:
        return {
            "final_score": 0,
            "sentiment": "neutral"
        }

    final_score = weighted_sum / weight_total

    if final_score > 0.2:
        sentiment = "positive"
    elif final_score < -0.2:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "final_score": round(final_score, 3),
        "sentiment": sentiment
    }