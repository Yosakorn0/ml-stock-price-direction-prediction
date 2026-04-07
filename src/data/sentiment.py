from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import finnhub
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

load_dotenv(".env.local")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        print(f"Loading sentiment model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        
        self.finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

    def fetch_news(self, symbol, days=7):
        """Fetch news for a specific symbol from Finnhub."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        print(f"Fetching news for {symbol} from {start_date} to {end_date}")
        res = self.finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        return res

    def analyze_texts(self, texts):
        """Analyze sentiment of a list of strings."""
        if not texts:
            return []
        
        # We can pass the whole list to the pipeline
        results = self.nlp(texts)
        return results

    def get_daily_sentiment(self, symbol, days=7):
        """Fetch and aggregate sentiment for a symbol."""
        news = self.fetch_news(symbol, days=days)
        if not news:
            return pd.DataFrame()

        df_news = pd.DataFrame(news)
        df_news['datetime'] = pd.to_datetime(df_news['datetime'], unit='s')
        
        headlines = df_news['headline'].tolist()
        sentiments = self.analyze_texts(headlines)
        
        df_news['sentiment'] = [s['label'] for s in sentiments]
        df_news['score'] = [s['score'] for s in sentiments]
        
        # Map sentiment labels to scores: positive=1, neutral=0, negative=-1
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        df_news['sentiment_val'] = df_news['sentiment'].map(sentiment_map)
        
        # Aggregate by day
        daily_sentiment = df_news.groupby(df_news['datetime'].dt.date)['sentiment_val'].mean()
        return daily_sentiment

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test for MSFT
    print("\nTesting sentiment analysis for MSFT...")
    sentiment = analyzer.get_daily_sentiment("MSFT", days=3)
    print("Daily Sentiment Average:")
    print(sentiment)
