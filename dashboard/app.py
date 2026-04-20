import streamlit as st

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Vortex | Financial Intelligence Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import plotly.graph_objects as go
import os
import sys
from datetime import datetime

# Add project root to path to fix ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predict import Predictor
import time
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# --- PROMETHEUS METRICS ---
# Initialize metrics once using Streamlit's cache_resource
@st.cache_resource
def get_metrics():
    try:
        start_http_server(8000, addr='0.0.0.0')
    except Exception:
        pass # Server already running
        
    return {
        'count': Counter('vortex_predictions', 'Total number of stock predictions made', ['symbol', 'prediction']),
        'latency': Histogram('vortex_inference_latency', 'Time spent processing model inference'),
        'sentiment': Gauge('vortex_market_sentiment', 'Latest analyzed sentiment score', ['symbol']),
        'errors': Counter('vortex_errors', 'Total number of system failures', ['type', 'symbol'])
    }

metrics = get_metrics()
PREDICTION_COUNT = metrics['count']
INFERENCE_TIME = metrics['latency']
SENTIMENT_VAL = metrics['sentiment']
ERROR_COUNT = metrics['errors']

# --- PREMIUM STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #16213e, #0f3460);
    }

    /* Glassmorphism Containers */
    div[data-testid="stMetricValue"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }

    div[data-testid="stMetricValue"]:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.3);
    }

    /* Sidebar Background */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(15px);
    }

    /* Custom Header */
    .main-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    /* Prediction Result Cards */
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1rem;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stMarkdown, .stPlotlyChart {
        animation: fadeIn 0.8s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<h1 class="main-header">Vortex Intelligence</h1>', unsafe_allow_html=True)
st.markdown("##### *Real-time Multi-Source Financial Forecasting System*")
st.write("")

# Initialize Predictor
@st.cache_resource
def get_predictor():
    try:
        return Predictor()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None

predictor = get_predictor()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/plasticine/200/financial-growth-analysis.png", width=100)
    st.header("Asset Portfolio")
    
    symbols = {
        'MSFT': 'Microsoft Corp',
        'AMZN': 'Amazon.com Inc',
        'GOOGL': 'Alphabet Inc',
        'GC=F': 'Gold Futures',
        'BTC-USD': 'Bitcoin'
    }
    
    selected_symbol = st.selectbox(
        "Select Target Asset:", 
        list(symbols.keys()), 
        format_func=lambda x: symbols[x]
    )
    
    st.markdown("---")
    st.subheader("🛠️ System Integrity")
    status_col1, status_col2 = st.columns(2)
    status_col1.caption("Model Version")
    status_col1.write("**XGB-2.1.wfv**")
    status_col2.caption("Data Provider")
    fh_key_loaded = os.getenv("FINNHUB_API_KEY") is not None
    if fh_key_loaded:
        status_col2.write("🟢 **AUTHORIZED**")
    else:
        status_col2.write("🟡 **LIMITED**")
    
    st.markdown("---")
    st.subheader("📉 Backtest Validation")
    st.caption("Out-of-Sample Sharpe (Net of 0.1% Friction)")
    
    performance_data = {
        "Asset": ["Gold", "MSFT", "GOOGL", "AMZN", "BTC"],
        "Sharpe": [1.49, -0.26, -0.01, -0.71, -0.78]
    }
    perf_df = pd.DataFrame(performance_data)
    
    def highlight_sharpe(val):
        color = '#00f2fe' if val > 0 else '#ff4b4b'
        return f'color: {color}; font-weight: bold'

    st.table(perf_df.style.map(highlight_sharpe, subset=['Sharpe']))
    
    st.info("Core Engine: FinBERT + XGBoost Ensemble")

# --- MAIN DASHBOARD LOGIC ---
if st.button("Generate Intelligence Report", use_container_width=True):
    if not predictor:
        st.error("Predictor engine not initialized. Check your logs.")
    else:
        with st.spinner(f"Synthesizing data for {symbols[selected_symbol]}..."):
            start_time = time.time()
            result = predictor.predict_direction(selected_symbol)
            latency = time.time() - start_time
            INFERENCE_TIME.observe(latency)
            
            if "error" in result:
                error_type = "Market Data" if "Market Data" in result['error'] else "Engine"
                ERROR_COUNT.labels(type=error_type, symbol=selected_symbol).inc()
                
                if "Market Data" in result['error']:
                    st.warning(f"🕒 **Data Latency Detected:** {result['error']}")
                    st.info("This often happens during market holidays or due to provider rate limits. Please try again in a few minutes.")
                else:
                    st.error(f"⚠️ **Engine failure:** {result['error']}")
            else:
                # Record Telemetry
                PREDICTION_COUNT.labels(symbol=selected_symbol, prediction=result['prediction']).inc()
                SENTIMENT_VAL.labels(symbol=selected_symbol).set(result['sentiment'])
                
                # Top Metrics
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.metric("Latest Market Price", f"${result['latest_price']:,.2f}")
                
                with m2:
                    pred_color = "inverse" if result['prediction'] == "DOWN" else "normal"
                    st.metric("Directional Forecast", result['prediction'], delta=None, delta_color=pred_color)
                
                with m3:
                    st.metric("Model Confidence", f"{result['probability']:.1%}")
                
                with m4:
                    sentiment_label = "Bullish" if result['sentiment'] > 0.1 else ("Bearish" if result['sentiment'] < -0.1 else "Neutral")
                    st.metric("Market Sentiment", sentiment_label, delta=f"{result['sentiment']:.2f}")

                st.markdown("---")
                
                # Visual Breakdown
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.subheader("📈 Momentum & Historical Convergence")
                    clean_symbol = selected_symbol.replace("=F", "_GOLD").replace("-USD", "_BTC")
                    hist_path = f"data/processed/{clean_symbol}_features.csv"
                    
                    if os.path.exists(hist_path):
                        df_hist = pd.read_csv(hist_path, index_col=0, parse_dates=True)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_hist.index[-60:], 
                            y=df_hist['Close'][-60:], 
                            name='Price',
                            line=dict(color='#00f2fe', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(0, 242, 254, 0.1)'
                        ))
                        fig.update_layout(
                            template="plotly_dark",
                            margin=dict(l=20, r=20, t=20, b=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Execute 'train.py' pipeline to populate historical features.")

                with col_right:
                    st.subheader("🔍 Feature Analytics")
                    st.write(f"**Relative Strength (RSI):** `{result['rsi']:.2f}`")
                    
                    rsi_status = "Overbought" if result['rsi'] > 70 else ("Oversold" if result['rsi'] < 30 else "Neutral")
                    st.progress(min(max(result['rsi']/100, 0.0), 1.0))
                    st.caption(f"Oscillator State: {rsi_status}")
                    
                    st.markdown("---")
                    st.write("**NLP Core Signals:**")
                    st.caption("Aggregated from recent headlines via FinBERT")
                    st.write(f"Processed at: `{datetime.now().strftime('%H:%M:%S')}`")

else:
    # Empty State
    st.markdown("""
    <div style="text-align: center; padding: 5rem; opacity: 0.6;">
        <img src="https://img.icons8.com/fluency/96/search-database.png" />
        <h2 style="color: #4facfe;">Awaiting Asset Configuration</h2>
        <p>Select an asset from the sidebar and click 'Generate Intelligence Report' to begin.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("⚠️ **Disclaimer:** This system is for academic research only. Predicted accuracy is based on backtest data and does not guarantee future results.")
