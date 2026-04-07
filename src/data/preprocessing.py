import pandas as pd
import numpy as np
import os
from src.data.ingestion import DataIngestion
from src.data.sentiment import SentimentAnalyzer
from src.features.technical import TechnicalFeatures

class Preprocessor:
    def __init__(self, symbols=None):
        self.symbols = symbols or ['MSFT', 'AMZN', 'GOOGL', 'GC=F', 'BTC-USD']
        self.ingestor = DataIngestion()
        self.analyzer = SentimentAnalyzer()
        self.tech_features = TechnicalFeatures()

    def process_asset(self, symbol):
        """Full pipeline for a single asset."""
        print(f"\nProcessing all features for {symbol}...")
        
        # 1. Fetch market data
        market_data = self.ingestor.fetch_market_data([symbol])
        
        # Handle MultiIndex if necessary
        if isinstance(market_data.columns, pd.MultiIndex):
            market_data.columns = market_data.columns.get_level_values(0)
            
        # 2. Add Technical Features
        df = self.tech_features.add_all_features(market_data)
        
        # 3. Add Sentiment (Last 30 days for demonstration/test)
        # Note: In a production env, sentiment would be fetched daily/stored
        print(f"Fetching sentiment for {symbol}...")
        sentiment = self.analyzer.get_daily_sentiment(symbol, days=30)
        
        if not sentiment.empty:
            df['Sentiment'] = df.index.map(lambda x: sentiment.get(x.date(), 0))
            df['Sentiment'] = df['Sentiment'].fillna(0) # Default to neutral
        else:
            df['Sentiment'] = 0
            
        # 4. Create Target (1 if Price goes up tomorrow, else 0)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        return df.dropna()

    def prepare_all_data(self):
        """Prepare data for all symbols and save."""
        os.makedirs("data/processed", exist_ok=True)
        
        # Fetch shared macro data once
        macro_data = self.ingestor.fetch_macro_data()
        macro_daily = macro_data.resample('D').pad()
        
        combined_data = {}
        for symbol in self.symbols:
            df = self.process_asset(symbol)
            
            # Merge with macro data
            df = df.join(macro_daily, how='left')
            df = df.fillna(method='ffill')
            
            # Save individual symbol data
            clean_symbol = symbol.replace("=F", "_GOLD").replace("-USD", "_BTC")
            save_path = f"data/processed/{clean_symbol}_features.csv"
            df.to_csv(save_path)
            print(f"Saved processed data for {symbol} to {save_path}")
            combined_data[symbol] = df
            
        return combined_data

if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.prepare_all_data()
