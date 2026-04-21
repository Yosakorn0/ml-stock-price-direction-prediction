import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

def generate_charts():
    processed_dir = "data/processed"
    model_dir = "models"
    fig_dir = "images"
    os.makedirs(fig_dir, exist_ok=True)
    
    symbols = {
        'MSFT': 'MSFT_features.csv',
        'GOLD': 'GC_GOLD_features.csv',
        'BTC': 'BTC_BTC_features.csv'
    }
    
    print("Generating SHAP Significance Charts...")
    for symbol, filename in symbols.items():
        feat_path = os.path.join(processed_dir, filename)
        model_name = filename.replace("_features.csv", "_xgb.joblib")
        mod_path = os.path.join(model_dir, model_name)
        
        if not os.path.exists(mod_path):
            print(f"Skipping {symbol}: Model not found.")
            continue
            
        print(f"Processing {symbol}...")
        df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
        model = joblib.load(mod_path)
        X = df.drop(columns=['Target'])
        
        # Explain the last 50 days (Test set representative)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.tail(50))
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X.tail(50), show=False)
        plt.title(f"Model Intelligence: Feature Impact ({symbol})")
        
        save_path = f"{fig_dir}/shap_summary_{symbol}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    generate_charts()
