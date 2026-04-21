import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

processed_dir = "data/processed"
model_dir = "models"
files = {
    'MSFT': 'MSFT_features.csv',
    'AMZN': 'AMZN_features.csv',
    'GOOGL': 'GOOGL_features.csv',
    'GOLD': 'GC_GOLD_features.csv',
    'BTC': 'BTC_BTC_features.csv'
}

print("Accuracy Report (Out-of-Sample):\n" + "="*35)
for asset, filename in files.items():
    feat_path = os.path.join(processed_dir, filename)
    model_name = filename.replace("_features.csv", "_xgb.joblib")
    mod_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(mod_path):
        print(f"{asset}: Model not found")
        continue
        
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    model = joblib.load(mod_path)
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    split_idx = int(len(df) * 0.8)
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{asset}: {acc:.2%}")
