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
    # Note: Modern yfinance often prefers its own session management 
    # or curl_cffi for bypassing anti-bot measures.
    import requests
    session = requests.Session()
    session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
    return session

load_dotenv(".env.local")

# Authenticated Clients
fh_key = os.getenv("FINNHUB_API_KEY")
if fh_key:
    print("Priority A: Finnhub Key detected. Authenticated ingestion enabled.")
    FINNHUB_CLIENT = finnhub.Client(api_key=fh_key)
else:
    print("Warning: FINNHUB_API_KEY missing. App running in unauthenticated 'Public' mode.")
    FINNHUB_CLIENT = None

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
        self.last_status = [] # For diagnostic reporting
        
    def fetch_finnhub_data(self, symbol):
        """Authenticated fetch using Finnhub API."""
        try:
            import time
            msg = f"Strategy 0 (Finnhub): Attempting {symbol}"
            print(msg)
            self.last_status.append(msg)
            
            # Map symbol for Finnhub
            fh_symbol = symbol.replace("-USD", "USDT")
            if "USDT" in fh_symbol and ":" not in fh_symbol:
                fh_symbol = f"BINANCE:{fh_symbol}"
            elif symbol == "GC=F":
                # Gold is tricky in Finnhub; try OANDA or just GC=F
                fh_symbol = "OANDA:XAU_USD"
            
            to_ts = int(time.time())
            from_ts = to_ts - (90 * 24 * 3600)
            
            res = FINNHUB_CLIENT.stock_candles(fh_symbol, 'D', from_ts, to_ts)
            
            # Fallback for Gold if OANDA fails
            if res['s'] != 'ok' and symbol == "GC=F":
                fh_symbol = "GC=F"
                res = FINNHUB_CLIENT.stock_candles(fh_symbol, 'D', from_ts, to_ts)
            
            if res['s'] == 'ok':
                df = pd.DataFrame({
                    'Open': res['o'],
                    'High': res['h'],
                    'Low': res['l'],
                    'Close': res['c'],
                    'Volume': res['v']
                }, index=pd.to_datetime(res['t'], unit='s'))
                self.last_status.append(f"Strategy 0 (Finnhub): SUCCESS ({len(df)} rows)")
                return df
            else:
                self.last_status.append(f"Strategy 0 (Finnhub): FAILED status={res['s']}")
        except Exception as e:
            self.last_status.append(f"Strategy 0 (Finnhub): ERROR {str(e)}")
        return pd.DataFrame()

    def fetch_alpha_vantage_data(self, symbol):
        """Authenticated fetch using Alpha Vantage API."""
        try:
            msg = f"Strategy 1 (AlphaVantage): Attempting {symbol}"
            print(msg)
            self.last_status.append(msg)
            
            av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            if not av_key:
                self.last_status.append("Strategy 1 (AlphaVantage): SKIP (No Key)")
                return pd.DataFrame()
            
            # Specific Handling for Crypto vs Stocks
            is_crypto = "-USD" in symbol or symbol == "BTC"
            if is_crypto:
                av_symbol = symbol.replace("-USD", "")
                url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={av_symbol}&market=USD&apikey={av_key}'
            else:
                if symbol == "GC=F":
                    # Skip AV for Gold Futures as it often returns incorrect mappings
                    self.last_status.append("Strategy 1 (AlphaVantage): SKIP (Inconsistent Gold Mapping)")
                    return pd.DataFrame()
                
                av_symbol = symbol
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_symbol}&apikey={av_key}&outputsize=compact'
                
            r = requests.get(url, timeout=10)
            data = r.json()
            
            # Parse Based on Endpoint
            if is_crypto and "Time Series (Digital Currency Daily)" in data:
                ts_data = data["Time Series (Digital Currency Daily)"]
                df = pd.DataFrame.from_dict(ts_data, orient='index')
                # Correct cols for AlphaVantage Crypto: '4a. close (USD)'
                df = df[['1a. open (USD)', '2a. high (USD)', '3a. low (USD)', '4a. close (USD)', '5. volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            elif not is_crypto and "Time Series (Daily)" in data:
                ts_data = data["Time Series (Daily)"]
                df = pd.DataFrame.from_dict(ts_data, orient='index')
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            else:
                note = data.get('Note', data.get('Error Message', data.get('Information', 'Unknown Error')))
                self.last_status.append(f"Strategy 1 (AlphaVantage): FAILED msg={note[:50]}")
                return pd.DataFrame()

            df = df.astype(float)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            self.last_status.append(f"Strategy 1 (AlphaVantage): SUCCESS ({len(df)} rows)")
            return df
        except Exception as e:
            self.last_status.append(f"Strategy 1 (AlphaVantage): ERROR {str(e)}")
        return pd.DataFrame()

    def fetch_market_data(self, symbols=None):
        """Fetch market data with ultra-robust fallback mechanisms."""
        self.last_status = [] # Reset for new request
        symbols = symbols or list(SYMBOLS.keys())
        import time
        
        # Strategy 0: Finnhub (Authenticated)
        if len(symbols) == 1:
            try:
                fh_data = self.fetch_finnhub_data(symbols[0])
                if not fh_data.empty and fh_data['Close'].dropna().shape[0] >= 2:
                    self._save_evergreen(symbols[0], fh_data)
                    return fh_data
            except Exception as e:
                print(f"Strategy 0 failed: {e}")

        # Strategy 1: Alpha Vantage (Authenticated)
        if len(symbols) == 1:
            try:
                av_data = self.fetch_alpha_vantage_data(symbols[0])
                if not av_data.empty and av_data['Close'].dropna().shape[0] >= 2:
                    self._save_evergreen(symbols[0], av_data)
                    return av_data
            except Exception as e:
                print(f"Strategy 1 failed: {e}")

        # Strategy A: Use single-ticker string (Often bypasses list-based blocking)
        if len(symbols) == 1:
            try:
                s = symbols[0]
                print(f"Strategy A: Ticker.history (string) for {s}")
                # Removed custom session as it conflicts with modern yf
                ticker = yf.Ticker(s)
                data = ticker.history(period="1y", interval="1d", auto_adjust=True)
                if not data.empty and data['Close'].dropna().shape[0] >= 2:
                    data.index = data.index.tz_localize(None)
                    self._save_evergreen(s, data)
                    return data
            except Exception as e:
                print(f"Strategy A failed: {e}")
            
            time.sleep(1) # Small delay to reset connection

        # Strategy B: yf.download (Standard)
        try:
            print(f"Strategy B: yf.download for {symbols}")
            data = yf.download(symbols, period="1y", interval="1d", progress=False)
            if not data.empty:
                # Standardize index to be TZ-Naive
                data.index = data.index.tz_localize(None)
                
                # Basic check for at least 2 rows of data
                if isinstance(data.columns, pd.MultiIndex):
                    self._save_evergreen(symbols, data)
                    return data
                elif data['Close'].dropna().shape[0] >= 2:
                    self._save_evergreen(symbols, data)
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
                stooq_symbol = s.replace("-USD", "USD").replace("=F", ".F")
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
            
        except Exception as e:
            print(f"Strategy D failed: {e}")

        # Strategy E: 5-Day Pulse (Yahoo Short-term) - Last resort scraping
        try:
            print(f"Strategy E: yf.download (5-Day Pulse) for {symbols}")
            data = yf.download(symbols, period="5d", interval="1d", progress=False)
            if not data.empty and (isinstance(data.columns, pd.MultiIndex) or data['Close'].dropna().shape[0] >= 2):
                data.index = data.index.tz_localize(None)
                self._save_evergreen(symbols, data)
                return data
        except Exception as e:
            self.last_status.append(f"Strategy E failed: {e}")

        # Strategy F: Evergreen Cache (The Presentation Saver)
        print(f"Strategy F: Attempting Evergreen Cache for {symbols}")
        cached_data = self._load_evergreen(symbols)
        if not cached_data.empty:
            self.last_status.append(f"Strategy F (Evergreen): SUCCESS (Loaded from cache)")
            return cached_data

        return pd.DataFrame()

    def _save_evergreen(self, symbols, data):
        """Save a snapshot of successful data to be used during outages."""
        try:
            cache_dir = "data/raw/evergreen"
            os.makedirs(cache_dir, exist_ok=True)
            
            # If multi-index, we save per-asset
            if isinstance(data.columns, pd.MultiIndex):
                for symbol in (symbols if isinstance(symbols, list) else [symbols]):
                    asset_df = data.xs(symbol, axis=1, level=1) if symbol in data.columns.get_level_values(1) else data
                    asset_df.to_csv(f"{cache_dir}/{symbol.replace('=F', '_GOLD').replace('-USD', '_BTC')}.csv")
            else:
                symbol = symbols[0] if isinstance(symbols, list) else symbols
                data.to_csv(f"{cache_dir}/{symbol.replace('=F', '_GOLD').replace('-USD', '_BTC')}.csv")
        except Exception as e:
            print(f"Failed to save evergreen cache: {e}")

    def _load_evergreen(self, symbols):
        """Load data from the evergreen cache."""
        try:
            cache_dir = "data/raw/evergreen"
            symbol = symbols[0] if isinstance(symbols, list) else symbols
            clean_symbol = symbol.replace('=F', '_GOLD').replace('-USD', '_BTC')
            cache_path = f"{cache_dir}/{clean_symbol}.csv"
            
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                # Ensure TZ-Naive even when loading from old cache
                df.index = pd.to_datetime(df.index).tz_localize(None)
                # Flag this data as cached for the UI
                df.attrs['is_cached'] = True
                return df
        except Exception as e:
            print(f"Failed to load evergreen cache: {e}")
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
