import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Add project root to path to fix ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predict import Predictor

# Page configuration
st.set_page_config(page_title="Financial Intelligence Dashboard", layout="wide")

st.title("🌟 Multi-Source Financial Intelligence System")
st.markdown("---")

# Initialize Predictor
@st.cache_resource
def get_predictor():
    return Predictor()

predictor = get_predictor()

# Sidebar for Asset Selection
st.sidebar.header("Asset Selection")
symbols = {
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'GOOGL': 'Google',
    'GC=F': 'Gold',
    'BTC-USD': 'Bitcoin'
}
selected_symbol = st.sidebar.selectbox("Choose an asset:", list(symbols.keys()), format_func=lambda x: symbols[x])

# Scientific Backtest Section
st.sidebar.markdown("---")
st.sidebar.subheader("📉 Scientific Backtest")
st.sidebar.caption("Out-of-Sample Performance (Incl. 0.1% Friction)")

performance_data = {
    "Asset": ["Gold", "MSFT", "GOOGL", "AMZN", "BTC"],
    "OOS Sharpe": [1.49, -0.26, -0.01, -0.71, -0.78]
}
perf_df = pd.DataFrame(performance_data)

# Styling the dataframe
def highlight_sharpe(val):
    color = 'green' if val > 0 else 'red'
    return f'color: {color}'

st.sidebar.table(perf_df.style.applymap(highlight_sharpe, subset=['OOS Sharpe']))

st.sidebar.info("Model: XGBoost Ensemble with 10bps Transaction Costs")

if st.sidebar.button("Fetch Latest Prediction"):
    with st.spinner(f"Analyzing {selected_symbol}..."):
        result = predictor.predict_direction(selected_symbol)
        
        if "error" in result:
            st.error(result["error"])
        else:
            # Layout Columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Latest Price", f"${result['latest_price']:.2f}")
                st.write(f"Updated: {result['timestamp']}")
                
            with col2:
                color = "green" if result['prediction'] == "UP" else "red"
                st.markdown(f"### Next-Day Prediction: <span style='color:{color}'>{result['prediction']}</span>", unsafe_allow_html=True)
                st.write(f"Confidence: {result['probability']:.2%}")
                
            with col3:
                # Sentiment Gauge
                sentiment = result['sentiment']
                st.markdown(f"### Sentiment Score: {sentiment:.2f}")
                if sentiment > 0.1:
                    st.success("Bullish Sentiment")
                elif sentiment < -0.1:
                    st.error("Bearish Sentiment")
                else:
                    st.warning("Neutral Sentiment")

            # Technical Indicators Section
            st.markdown("---")
            st.subheader("Technical Analysis Breakdown")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.write(f"RSI: {result['rsi']:.2f}")
                if result['rsi'] > 70:
                    st.write("Condition: Overbought")
                elif result['rsi'] < 30:
                    st.write("Condition: Oversold")
                else:
                    st.write("Condition: Neutral")

            # Historical Chart (Placeholders for real historical data if available)
            st.markdown("---")
            st.subheader("Price History")
            # For a real implementation, we'd load the full processed DataFrame here
            clean_symbol = selected_symbol.replace("=F", "_GOLD").replace("-USD", "_BTC")
            hist_path = f"data/processed/{clean_symbol}_features.csv"
            if os.path.exists(hist_path):
                df_hist = pd.read_csv(hist_path, index_col=0, parse_dates=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Close'], name='Close Price'))
                fig.update_layout(title=f"{symbols[selected_symbol]} Historical Price", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run the training pipeline or 'Fetch Latest Prediction' to see historical trends.")

st.sidebar.markdown("---")
st.sidebar.info("This system uses FinBERT for sentiment analysis and XGBoost for classification.")
