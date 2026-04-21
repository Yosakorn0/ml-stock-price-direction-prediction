import joblib
import pandas as pd
import numpy as np
import os
from src.data.preprocessing import Preprocessor, DataFetchError

class Predictor:
    def __init__(self, model_dir=None):
        # Dynamically find the project root to avoid path issues
        if model_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.model_dir = os.path.join(base_dir, "models")
        else:
            self.model_dir = model_dir
            
        self.preprocessor = Preprocessor()

    def get_latest_data(self, symbol):
        """Fetch and process the most recent data for a symbol."""
        # This can now raise DataFetchError
        df = self.preprocessor.process_asset(symbol, include_macro=True)
        return df.tail(1)

    def predict_direction(self, symbol):
        """Predict the next-day direction for the given symbol."""
        try:
            clean_symbol = symbol.replace("=F", "_GOLD").replace("-USD", "_BTC")
            model_path = f"{self.model_dir}/{clean_symbol}_xgb.joblib"
            feature_path = f"{self.model_dir}/{clean_symbol}_features.joblib"
            
            if not os.path.exists(model_path):
                return {"error": f"Intelligence Engine: Model for {symbol} not yet configured."}
            
            # Get latest data (Handling DataFetchError)
            latest_df = self.get_latest_data(symbol)
            
            if latest_df.empty:
                return {"error": f"Market Data: Results for {symbol} are currently empty. Please try again later."}

            # Load model and features
            model = joblib.load(model_path)
            expected_features = joblib.load(feature_path)
            
            X = latest_df[expected_features]
            
            # Predict
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            
            # Get latest price and sentiment
            latest_row = latest_df.iloc[-1]
            
            return {
                "symbol": symbol,
                "prediction": "UP" if pred == 1 else "DOWN",
                "probability": max(prob),
                "latest_price": latest_row['Close'],
                "sentiment": latest_row.get('Sentiment', 0),
                "rsi": latest_row.get('RSI', 0),
                "timestamp": latest_df.index[-1].strftime("%Y-%m-%d"),
                "is_cached": latest_df.attrs.get('is_cached', False)
            }
        except DataFetchError as e:
            return {"error": f"Market Data: {str(e)}"}
        except Exception as e:
            return {"error": f"Internal Engine Error: {str(e)}"}

if __name__ == "__main__":
    predictor = Predictor()
    # Example prediction for MSFT
    result = predictor.predict_direction("MSFT")
    print("\nLatest Prediction Result:")
    print(result)
