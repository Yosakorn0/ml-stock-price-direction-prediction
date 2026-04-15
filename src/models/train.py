import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import shap
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, processed_dir="data/processed", model_dir="models", fig_dir="images"):
        self.processed_dir = processed_dir
        self.model_dir = model_dir
        self.fig_dir = fig_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.fig_dir, exist_ok=True)

    def load_data(self, symbol):
        """Load processed feature data for a symbol."""
        clean_symbol = symbol.replace("=F", "_GOLD").replace("-USD", "_BTC")
        file_path = f"{self.processed_dir}/{clean_symbol}_features.csv"
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df

    def calculate_sharpe_ratio(self, y_test, y_pred, returns_series, transaction_cost=0.001):
        """
        Calculate the annualized Sharpe Ratio for a directional strategy.
        Strategy: Long if prediction is UP (1), Short if prediction is DOWN (0).
        """
        # Convert binary predictions to signals (-1, 1)
        signals = np.where(y_pred == 1, 1, -1)
        
        # Calculate strategy returns
        # strategy_return = signal_{t} * return_{t}
        strategy_returns = signals * returns_series
        
        # Apply transaction costs (only when signal changes)
        # simplified: apply cost to every trade entry/reversal
        signal_changes = np.diff(signals, prepend=0) != 0
        strategy_returns[signal_changes] -= transaction_cost
        
        avg_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns)
        
        if std_ret == 0:
            return 0
            
        sharpe = (avg_ret / std_ret) * np.sqrt(252) # Annualized
        return sharpe

    def train_symbol_model(self, symbol, use_tuning=True):
        """Train and save an optimized XGBoost model for a specific symbol."""
        df = self.load_data(symbol)
        if df is None:
            return
        
        print(f"\nTraining pipeline for {symbol}...")
        
        # 1. Prepare Features (X) and Target (y)
        X = df.drop(columns=['Target'])
        y = df['Target']
        
        # Save returns for Sharpe calculation later
        # Note: Daily_Return is expected in the features
        returns = df['Daily_Return'] if 'Daily_Return' in df.columns else df['Close'].pct_change()
        
        # Time-series split: we use the last 20% for testing (Out-of-Sample)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        returns_test = returns.iloc[split_idx:]
        
        # 2. Hyperparameter Tuning (Automated Grid/Random Search)
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        if use_tuning:
            print(f"Finding optimal hyperparameters for {symbol}...")
            search = RandomizedSearchCV(base_model, param_distributions=param_dist, 
                                        n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(f"Best Params: {search.best_params_}")
        else:
            model = base_model
            model.fit(X_train, y_train)
        
        # 3. Scientific Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        sharpe = self.calculate_sharpe_ratio(y_test, y_pred, returns_test)
        
        print(f"--- Scientific Metrics for {symbol} ---")
        print(f"OOS Accuracy: {acc:.4f}")
        print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # 4. SHAP Interpretability Analysis
        print(f"Generating SHAP summary for {symbol}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        clean_symbol = symbol.replace("=F", "_GOLD").replace("-USD", "_BTC")
        plt.title(f"SHAP Feature Importance: {symbol}")
        plt.tight_layout()
        plt.savefig(f"{self.fig_dir}/shap_summary_{clean_symbol}.png")
        plt.close()
        
        # 5. Save Artifacts
        model_path = f"{self.model_dir}/{clean_symbol}_xgb.joblib"
        joblib.dump(model, model_path)
        joblib.dump(X.columns.tolist(), f"{self.model_dir}/{clean_symbol}_features.joblib")
        print(f"Model and SHAP chart saved for {symbol}")

    def train_all(self, symbols=None):
        """Train models for all configured symbols."""
        symbols = symbols or ['MSFT', 'AMZN', 'GOOGL', 'GC=F', 'BTC-USD']
        for symbol in symbols:
            self.train_symbol_model(symbol)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all()
