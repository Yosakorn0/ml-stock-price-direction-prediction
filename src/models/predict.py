import joblib
import pandas as pd
import numpy as np
import os
from src.data.preprocessing import Preprocessor

class Predictor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.preprocessor = Preprocessor()

    def get_latest_data(self, symbol):
        """Fetch and process the most recent data for a symbol."""
        # For the latest prediction, we need the most recent features
        # preprocessing.py handles fetching and indicator calculation
        df = self.preprocessor.process_asset(symbol)
        return df.tail(1)

    def predict_direction(self, symbol):
        """Predict the next-day direction for the given symbol."""
        clean_symbol = symbol.replace("=F", "_GOLD").replace("-USD", "_BTC")
        model_path = f"{self.model_dir}/{clean_symbol}_xgb.joblib"
        feature_path = f"{self.model_dir}/{clean_symbol}_features.joblib"
        
        if not os.path.exists(model_path):
            return {"error": f"Model for {symbol} not found."}
        
        # Load model and features
        model = joblib.load(model_path)
        expected_features = joblib.load(feature_path)
        
        # Get latest data
        latest_df = self.get_latest_data(symbol)
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
            "timestamp": latest_df.index[-1].strftime("%Y-%m-%d")
        }

if __name__ == "__main__":
    predictor = Predictor()
    # Example prediction for MSFT
    result = predictor.predict_direction("MSFT")
    print("\nLatest Prediction Result:")
    print(result)
