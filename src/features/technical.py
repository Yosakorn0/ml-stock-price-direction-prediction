import pandas as pd
import numpy as np

class TechnicalFeatures:
    @staticmethod
    def calculate_rsi(data, window=14):
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(data, slow=26, fast=12, signal=9):
        """Calculate MACD."""
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_bollinger_bands(data, window=20, num_std=2):
        """Calculate Bollinger Bands."""
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def add_all_features(self, df):
        """Add all technical features to the DataFrame."""
        if df.empty:
            return df
            
        # Hard check for 'Close' column existence and validity
        if 'Close' not in df.columns or df['Close'].dropna().empty:
            return df
            
        # Assuming df has 'Close', 'High', 'Low', 'Volume' columns
        # If it's a MultiIndex (multiple assets), we apply per asset
        
        if isinstance(df.columns, pd.MultiIndex):
            # For each asset in the MultiIndex
            symbols = df.columns.get_level_values(1).unique()
            for symbol in symbols:
                df[('MA10_Dist', symbol)] = (df[('Close', symbol)] - df[('Close', symbol)].rolling(window=10).mean()) / df[('Close', symbol)].rolling(window=10).mean()
                df[('MA50_Dist', symbol)] = (df[('Close', symbol)] - df[('Close', symbol)].rolling(window=50).mean()) / df[('Close', symbol)].rolling(window=50).mean()
                df[('RSI', symbol)] = self.calculate_rsi(df[('Close', symbol)])
                df[('MACD', symbol)], _ = self.calculate_macd(df[('Close', symbol)])
                upper, lower = self.calculate_bollinger_bands(df[('Close', symbol)])
                df[('BB_Upper_Dist', symbol)] = (upper - df[('Close', symbol)]) / df[('Close', symbol)]
                df[('BB_Lower_Dist', symbol)] = (lower - df[('Close', symbol)]) / df[('Close', symbol)]
                df[('Daily_Return', symbol)] = df[('Close', symbol)].pct_change()
        else:
            df['MA10_Dist'] = (df['Close'] - df['Close'].rolling(window=10).mean()) / df['Close'].rolling(window=10).mean()
            df['MA50_Dist'] = (df['Close'] - df['Close'].rolling(window=50).mean()) / df['Close'].rolling(window=50).mean()
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['MACD'], _ = self.calculate_macd(df['Close'])
            upper, lower = self.calculate_bollinger_bands(df['Close'])
            df['BB_Upper_Dist'] = (upper - df['Close']) / df['Close']
            df['BB_Lower_Dist'] = (lower - df['Close']) / df['Close']
            df['Daily_Return'] = df['Close'].pct_change()
            
        return df.dropna()

if __name__ == "__main__":
    # Test with dummy data
    dates = pd.date_range("2023-01-01", periods=100)
    data = pd.DataFrame({'Close': np.random.randn(100).cumsum() + 100}, index=dates)
    
    tf = TechnicalFeatures()
    featured_data = tf.add_all_features(data)
    print("Featured Data Head:")
    print(featured_data.head())
    print("\nFeatures added successfully.")
