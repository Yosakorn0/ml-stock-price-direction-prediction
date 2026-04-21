from src.data.ingestion import DataIngestion
import pandas as pd
import unittest
from unittest.mock import patch

class TestIngestionFallback(unittest.TestCase):
    def test_evergreen_fallback(self):
        ingestor = DataIngestion()
        
        # We mock all methods that fetch data from external sources
        # to return empty dataframes, forcing Strategy F.
        with patch.object(DataIngestion, 'fetch_finnhub_data', return_value=pd.DataFrame()):
            with patch.object(DataIngestion, 'fetch_alpha_vantage_data', return_value=pd.DataFrame()):
                # Strategy A/B use yfinance which we mock at the library level or method level
                with patch('yfinance.download', return_value=pd.DataFrame()):
                    with patch('yfinance.Ticker') as mock_ticker:
                        mock_ticker.return_value.history.return_value = pd.DataFrame()
                        
                        print("\nTesting Fallback for MSFT...")
                        data = ingestor.fetch_market_data(['MSFT'])
                        
                        # Check result
                        if not data.empty and data.attrs.get('is_cached'):
                            print("SUCCESS: Fallback to Evergreen Cache works!")
                            print(f"Latest Cache Date: {data.index[-1]}")
                        else:
                            print("FAILURE: System did not fallback correctly.")
                            print(f"Data empty? {data.empty}")
                            print(f"Is cached? {data.attrs.get('is_cached', False)}")

if __name__ == "__main__":
    test = TestIngestionFallback()
    test.test_evergreen_fallback()
