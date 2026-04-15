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

def calculate_sharpe(file_path, model_path, transaction_cost=0.001):
    """Calculate Out-of-Sample Sharpe Ratio with transaction costs."""
    if not os.path.exists(model_path):
        return None
        
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    model = joblib.load(model_path)
    
    # Prepare features
    X = df.drop(columns=['Target'])
    
    # Split: only evaluate on the last 20% (Test Set)
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    df_test = df.iloc[split_idx:].copy()
    
    # Predict signals
    y_pred = model.predict(X_test)
    df_test['Signal'] = y_pred
    
    # Daily returns
    df_test['Returns'] = df_test['Close'].pct_change()
    
    # Strategy returns (Next-day return if signal is today)
    # Signal at T holds from T to T+1
    df_test['Raw_Strategy_Returns'] = df_test['Returns'].shift(-1) * df_test['Signal']
    
    # Transaction Costs: applied when signal changes
    # We shift Signal to find changes
    df_test['Trades'] = df_test['Signal'].diff().abs()
    # Fill first trade if Signal[0] == 1
    df_test.loc[df_test.index[0], 'Trades'] = df_test['Signal'].iloc[0]
    
    df_test['Cost'] = df_test['Trades'] * transaction_cost
    df_test['Net_Strategy_Returns'] = df_test['Raw_Strategy_Returns'] - df_test['Cost']
    
    strat_rets = df_test['Net_Strategy_Returns'].dropna()
    if len(strat_rets) == 0:
        return 0, 0
        
    # Annualized Sharpe (Assuming 252 trading days)
    # Annualized Return / Annualized Std
    ann_return = strat_rets.mean() * 252
    ann_vol = strat_rets.std() * np.sqrt(252)
    
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    return sharpe, ann_return

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
        sharpe_res = calculate_sharpe(feat_path, mod_path)
        
        print(f"\n{asset}:")
        print(f" - ADF Stat: {adf_res['adf_stat']:.4f} (p={adf_res['p_value']:.4f})")
        print(f" - Stationarity: {'Stationary' if adf_res['p_value'] < 0.05 else 'Non-Stationary'}")
        if sharpe_res:
            sharpe, ann_ret = sharpe_res
            print(f" - OOS Annualized Sharpe Ratio: {sharpe:.4f}")
            print(f" - OOS Annualized Return: {ann_ret:.2%}")
        else:
            print(" - Sharpe Ratio: Model not found")

if __name__ == "__main__":
    analyze()
