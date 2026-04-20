import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import os
from datetime import datetime
from dotenv import load_dotenv
import requests

import random
import finnhub

# Fix: Aggressive User-Agent rotation for cloud stability
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"
]

def get_session():
    session = requests.Session()
    session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
    return session

load_dotenv(".env.local")

# Authenticated Clients
FINNHUB_CLIENT = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

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
        
    def fetch_finnhub_data(self, symbol):
        """Authenticated fetch using Finnhub API."""
        try:
            import time
            print(f"Strategy 0: Finnhub (Authenticated) for {symbol}")
            
            # Map symbol for Finnhub
            fh_symbol = symbol.replace("-USD", "USDT")
            if "USDT" in fh_symbol and ":" not in fh_symbol:
                fh_symbol = f"BINANCE:{fh_symbol}"
            elif symbol == "GC=F":
                fh_symbol = "OANDA:XAU_USD"
            
            # 30 days back in unix timestamp
            to_ts = int(time.time())
            from_ts = to_ts - (365 * 24 * 3600)
            
            res = FINNHUB_CLIENT.stock_candles(fh_symbol, 'D', from_ts, to_ts)
            
            if res['s'] == 'ok':
                df = pd.DataFrame({
                    'Open': res['o'],
                    'High': res['h'],
                    'Low': res['l'],
                    'Close': res['c'],
                    'Volume': res['v']
                }, index=pd.to_datetime(res['t'], unit='s'))
                return df
        except Exception as e:
            print(f"Finnhub failed for {symbol}: {e}")
        return pd.DataFrame()

    def fetch_market_data(self, symbols=None):
        """Fetch market data with ultra-robust fallback mechanisms."""
        symbols = symbols or list(SYMBOLS.keys())
        import time
        
        # Strategy 0: Finnhub (Authenticated) - ONLY for single symbol requests
        if len(symbols) == 1:
            fh_data = self.fetch_finnhub_data(symbols[0])
            if not fh_data.empty and fh_data['Close'].dropna().shape[0] >= 2:
                return fh_data

        # Strategy A: Use single-ticker string (Often bypasses list-based blocking)
        if len(symbols) == 1:
            try:
                s = symbols[0]
                print(f"Strategy A: Ticker.history (string) for {s}")
                sess = get_session()
                ticker = yf.Ticker(s, session=sess)
                data = ticker.history(period="1y", interval="1d", auto_adjust=True)
                if not data.empty and data['Close'].dropna().shape[0] >= 2:
                    return data
            except Exception as e:
                print(f"Strategy A failed: {e}")
            
            time.sleep(1) # Small delay to reset connection

        # Strategy B: yf.download with randomized session
        try:
            print(f"Strategy B: yf.download (random session) for {symbols}")
            sess = get_session()
            data = yf.download(symbols, period="1y", interval="1d", progress=False, session=sess)
            if not data.empty:
                # Basic check for at least 2 rows of data
                if isinstance(data.columns, pd.MultiIndex):
                    return data
                elif data['Close'].dropna().shape[0] >= 2:
                    return data
        except Exception as e:
            print(f"Strategy B failed: {e}")

        time.sleep(1)

        # Strategy C: Raw yf.download (No custom session - uses library defaults)
        try:
            print(f"Strategy C: yf.download (Default settings) for {symbols}")
            data = yf.download(symbols, period="1y", interval="1d", progress=False)
            if not data.empty and (isinstance(data.columns, pd.MultiIndex) or data['Close'].dropna().shape[0] >= 2):
                return data
        except Exception as e:
            print(f"Strategy C failed: {e}")

        # Strategy D: Stooq (Fail-Safe) - Works when Yahoo is completely blocked
        try:
            print(f"Strategy D: Stooq Fail-Safe for {symbols}")
            import pandas_datareader.data as web
            all_stooq_data = []
            
            for s in (symbols if isinstance(symbols, list) else [symbols]):
                # Stooq doesn't use the same suffixes as Yahoo for some assets
                stooq_symbol = s.replace("-USD", "").replace("=F", "")
                try:
                    s_data = web.DataReader(stooq_symbol, 'stooq', start='2024-01-01')
                    if not s_data.empty:
                        # Stooq returns data in descending order, we need ascending
                        s_data = s_data.sort_index()
                        # Standardize columns to capitalized 
                        rename_map = {col: col.capitalize() for col in s_data.columns}
                        s_data = s_data.rename(columns=rename_map)
                        if len(symbols) == 1:
                            return s_data
                        all_stooq_data.append(s_data)
                except Exception as stooq_e:
                    print(f"Stooq failed for {stooq_symbol}: {stooq_e}")
            
            if all_stooq_data:
                # Basic combine if multiple symbols (simplified for this context)
                return pd.concat(all_stooq_data, axis=1) if len(all_stooq_data) > 1 else all_stooq_data[0]
                
        except Exception as e:
            print(f"Strategy D failed: {e}")

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
