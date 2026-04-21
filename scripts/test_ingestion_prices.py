from src.data.ingestion import DataIngestion
import pandas as pd

ingestor = DataIngestion()
for symbol in ['BTC-USD', 'GC=F', 'MSFT']:
    print(f"\nTesting Ingestion for {symbol}...")
    data = ingestor.fetch_market_data([symbol])
    if not data.empty:
        print(f"Latest Close: {data['Close'].iloc[-1]}")
        print(f"Index Head: {data.index[0]}")
        print(f"Index Tail: {data.index[-1]}")
    else:
        print("Failed to fetch data.")
    print("Logs:")
    for log in ingestor.last_status:
        print(f"  {log}")
