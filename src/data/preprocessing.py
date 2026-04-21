import pandas as pd
import numpy as np
import os
from src.data.ingestion import DataIngestion
from src.data.sentiment import SentimentAnalyzer
from src.features.technical import TechnicalFeatures

class DataFetchError(Exception):
    """Custom exception when market data fetch fails or returns empty."""
    pass

class Preprocessor:
    def __init__(self, symbols=None):
        self.symbols = symbols or ['MSFT', 'AMZN', 'GOOGL', 'GC=F', 'BTC-USD']
        self.ingestor = DataIngestion()
        self.analyzer = SentimentAnalyzer()
        self.tech_features = TechnicalFeatures()

    def process_asset(self, symbol, include_macro=False):
        """Full pipeline for a single asset."""
        print(f"\nProcessing all features for {symbol}...")
        
        # 1. Fetch market data
        market_data = self.ingestor.fetch_market_data([symbol])
        
        if market_data.empty or market_data['Close'].dropna().shape[0] < 2:
            raise DataFetchError(f"Insufficient market data for {symbol} (need at least 2 days of valid prices).")
            
        # Ensure TZ-Naive (Essential for Joining with Sentiment/Macro)
        if hasattr(market_data.index, 'tz') and market_data.index.tz is not None:
            market_data.index = market_data.index.tz_localize(None)
            
        # Handle MultiIndex if necessary
        if isinstance(market_data.columns, pd.MultiIndex):
            market_data.columns = market_data.columns.get_level_values(0)
            
        # 2. Add Technical Features
        df = market_data.copy()
        df = self.tech_features.add_all_features(df)
        
        # 3. Add Sentiment
        print(f"Fetching sentiment for {symbol}...")
        sentiment = self.analyzer.get_daily_sentiment(symbol, days=30)
        
        if not sentiment.empty:
            df.loc[:, 'Sentiment'] = df.index.map(lambda x: sentiment.get(x.date(), 0))
            df.loc[:, 'Sentiment'] = df['Sentiment'].fillna(0)
        else:
            df.loc[:, 'Sentiment'] = 0
            
        # 4. Create Target (1 if Price goes up tomorrow, else 0)
        df.loc[:, 'Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # 5. Optional Macro Data (Feature Consistency)
        if include_macro:
            macro_data = self.ingestor.fetch_macro_data()
            if not macro_data.empty:
                macro_daily = macro_data.resample('D').ffill()
                df = df.join(macro_daily, how='left')
                # Forward fill any gaps in macro data, then fill remaining with median or 0
                df = df.ffill().fillna(0)
            else:
                print(f"Warning: Macro data fetch failed for {symbol}. Continuing without macro features.")
            
        # Final cleanup: ONLY drop if core market price data is missing
        # We allow sentiment/macro to be 0/filled if missing, but we MUST have 'Close'
        final_df = df.dropna(subset=['Close'])
        
        # Ensure we still have Target for training mode
        if 'Target' in final_df.columns:
            final_df = final_df.dropna(subset=['Target'])

        # Preserve the is_cached attribute from market_data if it exists
        if hasattr(market_data, 'attrs') and market_data.attrs.get('is_cached'):
            final_df.attrs['is_cached'] = True
            
        return final_df

    def prepare_all_data(self):
        """Prepare data for all symbols and save."""
        os.makedirs("data/processed", exist_ok=True)
        
        combined_data = {}
        for symbol in self.symbols:
            # Join macro data for each symbol to ensure parity
            df = self.process_asset(symbol, include_macro=True)
            
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
