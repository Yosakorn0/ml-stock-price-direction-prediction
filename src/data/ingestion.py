import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import os
from datetime import datetime
from dotenv import load_dotenv
import requests

# Fix: Custom headers to prevent "No timezone found" / Bot protection in yfinance
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SESSION = requests.Session()
SESSION.headers.update({'User-Agent': USER_AGENT})

load_dotenv(".env.local")

SYMBOLS = {
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'GOOGL': 'Google',
    'GC=F': 'Gold',
    'BTC-USD': 'Bitcoin'
}

FRED_INDICATORS = ['FEDFUNDS', 'UNRATE', 'CPIAUCSL'] # Fed Funds, Unemployment, CPI

class DataIngestion:
    def __init__(self, start_date="2024-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
    def fetch_market_data(self, symbols=None):
        """Fetch market data with aggressive fallback mechanisms."""
        symbols = symbols or list(SYMBOLS.keys())
        
        # Strategy A: yf.download (Preferred for bulk and cloud stability)
        try:
            print(f"Strategy A: yf.download (session headers) for {symbols}")
            data = yf.download(symbols, period="1y", interval="1d", progress=False, session=SESSION, timeout=10)
            if not data.empty and data['Close'].dropna().shape[0] >= 2:
                return data
        except Exception as e:
            print(f"Strategy A failed: {e}")

        # Strategy B: yf.Ticker.history (Fallback for single symbols)
        if len(symbols) == 1:
            try:
                print(f"Strategy B: yf.Ticker.history (rotated headers) for {symbols[0]}")
                # Use a specific rotation of headers
                ALT_SESSION = requests.Session()
                ALT_SESSION.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'})
                ticker = yf.Ticker(symbols[0], session=ALT_SESSION)
                data = ticker.history(period="1y", interval="1d", auto_adjust=True, timeout=10)
                if not data.empty and data['Close'].dropna().shape[0] >= 2:
                    return data
            except Exception as e:
                print(f"Strategy B failed: {e}")

        # Strategy C: yf.download (NO custom session - let yfinance use internal defaults)
        try:
            print(f"Strategy C: yf.download (default library settings) for {symbols}")
            data = yf.download(symbols, period="1y", interval="1d", progress=False, timeout=15)
            if not data.empty and (isinstance(data.columns, pd.MultiIndex) or data['Close'].dropna().shape[0] >= 2):
                return data
        except Exception as e:
            print(f"Strategy C failed: {e}")

        # If all fail, return empty but let preprocessing handle the error
        return pd.DataFrame()

    def fetch_macro_data(self, indicators=None):
        """Fetch macroeconomic data from FRED."""
        indicators = indicators or FRED_INDICATORS
        print(f"Fetching macro data for: {indicators}")
        try:
            macro_data = web.DataReader(indicators, 'fred', self.start_date, self.end_date)
            return macro_data
        except Exception as e:
            print(f"Error fetching macro data: {e}")
            return pd.DataFrame()

    def merge_data(self, market_data, macro_data):
        """Merge market and macro data, aligning on date."""
        # Note: FRED data is often monthly/weekly. We forward fill to daily.
        macro_daily = macro_data.resample('D').ffill()
        
        # If market_data is MultiIndex (multiple symbols), we might want to merge per symbol
        # or just return both for now.
        return market_data, macro_daily

if __name__ == "__main__":
    ingestor = DataIngestion()
    market = ingestor.fetch_market_data()
    macro = ingestor.fetch_macro_data()
    
    print("Market Data Sample:")
    print(market.tail())
    print("\nMacro Data Sample:")
    print(macro.tail())
    
    # Save raw data
    os.makedirs("data/raw", exist_ok=True)
    market.to_csv("data/raw/market_data.csv")
    macro.to_csv("data/raw/macro_data.csv")
    print("\nData saved to data/raw/")
