import os
import finnhub
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv(".env.local")
fh_key = os.getenv("FINNHUB_API_KEY")
client = finnhub.Client(api_key=fh_key)

symbols = ["BINANCE:BTCUSDT", "OANDA:XAU_USD", "AMZN"]
to_ts = int(time.time())
from_ts = to_ts - (7 * 24 * 3600)

print("Finnhub Data Check:\n" + "="*30)
for s in symbols:
    print(f"\nFetching {s}...")
    res = client.stock_candles(s, 'D', from_ts, to_ts)
    if res['s'] == 'ok':
        print(f"Latest Close: {res['c'][-1]}")
    else:
        print(f"Failed: {res['s']}")
