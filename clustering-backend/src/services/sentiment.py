from transformers import pipeline
import feedparser
from dotenv import load_dotenv
import os
import requests
load_dotenv()

ticker = "META"
keyword = "meta"
news_api = os.getenv("NEWS_API_KEY")
date = "2025-01-25"

pipe = pipeline("text-classification", model="ProsusAI/finbert")

url =(
    'https://newsapi.org/v2/everything?'
    f'q={keyword}&'
    f'from={date}&'
    'sortBy=popularity&'
    f'apiKey={news_api}'
)

response = requests.get(url)
articles = response.json()['articles']
#print(articles)

articles = [article for article in articles if keyword.lower() in article["title"].lower() or keyword.lower() in article["description"].lower()]

#rss_url = f"https://www.finance.yahoo.com/rss/headline?s={ticker}"
#feed = feedparser.parse(rss_url)

total_score = 0
num_articles = 0


for i, article in enumerate(articles):
    #print(f"Title: {article["title"]}")
    #print(f"Published: {article["description"]}")
    #print(f"Link: {article["url"]}\n")
    
    sentiment = pipe(article["content"])[0]

    print(f"Sentiment: {sentiment['label']}, Pontszám: {sentiment['score']:.2f}")
    print("-" * 50)
    

    if sentiment["label"] == "positive":
        total_score += sentiment["score"]
        num_articles += 1
    elif sentiment["label"] == "negative":
        total_score -= sentiment["score"]
        num_articles += 1

final_score = total_score / num_articles
print(f'Overall: {"Positive" if final_score >= 0.15 else "Negative" if final_score <= -0.15 else "Neutral"} ({final_score:.2f})')
