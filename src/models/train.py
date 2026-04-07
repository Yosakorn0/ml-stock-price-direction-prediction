import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

class ModelTrainer:
    def __init__(self, processed_dir="data/processed", model_dir="models"):
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self, symbol):
        """Load processed feature data for a symbol."""
        clean_symbol = symbol.replace("=F", "_GOLD").replace("-USD", "_BTC")
        file_path = f"{self.processed_dir}/{clean_symbol}_features.csv"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    def train_symbol_model(self, symbol):
        """Train and save an XGBoost model for a specific symbol."""
        df = self.load_data(symbol)
        if df is None:
            return
        
        print(f"\nTraining model for {symbol}...")
        
        # Prepare Features (X) and Target (y)
        # Drop non-feature columns (Date is index, Target is y)
        X = df.drop(columns=['Target'])
        y = df['Target']
        
        # Time-series split: we use the last 20% for testing
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Define model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy for {symbol}: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        clean_symbol = symbol.replace("=F", "_GOLD").replace("-USD", "_BTC")
        model_path = f"{self.model_dir}/{clean_symbol}_xgb.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save feature names for inference
        joblib.dump(X.columns.tolist(), f"{self.model_dir}/{clean_symbol}_features.joblib")

    def train_all(self, symbols=None):
        """Train models for all configured symbols."""
        symbols = symbols or ['MSFT', 'AMZN', 'GOOGL', 'GC=F', 'BTC-USD']
        for symbol in symbols:
            self.train_symbol_model(symbol)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all()
