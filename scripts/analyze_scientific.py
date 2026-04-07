import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
import joblib

def calculate_adf(file_path):
    """Calculate ADF test for stationarity."""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # Testing Adjusted Close or Close for unit root
    res = adfuller(df['Close'].dropna())
    return {
        'adf_stat': res[0],
        'p_value': res[1],
        'lags': res[2],
        'n_obs': res[3],
        'crit_5': res[4]['5%']
    }

def calculate_sharpe(file_path, model_path):
    """Calculate Sharpe Ratio for the XGBoost strategy."""
    if not os.path.exists(model_path):
        return None
        
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    model = joblib.load(model_path)
    
    # Prepare features (same as train.py)
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    # Simple strategy: If pred=1, long; else, stay flat (or short, but let's do long/flat)
    y_pred = model.predict(X)
    
    # Daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # Strategy returns (Assume next-day return if signal is today)
    # Note: Shift(1) because signal at T predicts T+1
    df['Strategy_Returns'] = df['Returns'].shift(-1) * y_pred
    
    strat_rets = df['Strategy_Returns'].dropna()
    if len(strat_rets) == 0:
        return 0
        
    # Annualized Sharpe (Assuming 252 trading days)
    # Assuming 0% risk-free rate for simplicity
    sharpe = np.sqrt(252) * (strat_rets.mean() / strat_rets.std())
    return sharpe

def analyze():
    processed_dir = "data/processed"
    model_dir = "models"
    files = {
        'MSFT': 'MSFT_features.csv',
        'AMZN': 'AMZN_features.csv',
        'GOOGL': 'GOOGL_features.csv',
        'GOLD': 'GC_GOLD_features.csv',
        'BTC': 'BTC_BTC_features.csv'
    }
    
    print("Scientific Analysis Summary:\n" + "="*30)
    for asset, filename in files.items():
        feat_path = f"{processed_dir}/{filename}"
        model_name = filename.replace("_features.csv", "_xgb.joblib")
        mod_path = f"{model_dir}/{model_name}"
        
        # ADF Test
        adf_res = calculate_adf(feat_path)
        
        # Sharpe Ratio
        sharpe = calculate_sharpe(feat_path, mod_path)
        
        print(f"\n{asset}:")
        print(f" - ADF Stat: {adf_res['adf_stat']:.4f} (p={adf_res['p_value']:.4f})")
        print(f" - Stationarity: {'Stationary' if adf_res['p_value'] < 0.05 else 'Non-Stationary'}")
        if sharpe:
            print(f" - Annualized Sharpe Ratio: {sharpe:.4f}")
        else:
            print(" - Sharpe Ratio: Model not found")

if __name__ == "__main__":
    analyze()
