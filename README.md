# Cloud-Native Multi-Source Financial Intelligence System

**Author**: Yosakorn Sirisoot
**Academic Context**:

- **AT82.03**: Machine Learning
- **AT82.9002**: Selected Topic: Data Engineering and MLOps at AIT

## 🌟 Project Overview

This project develops an end-to-end machine learning pipeline to predict the direction of financial markets (Stocks, Forex, Gold) by integrating heterogeneous data sources: historical price data, technical indicators, fundamental indicators, macroeconomic variables, and sentiment information derived from news and social media.

The system builds a robust financial data engineering pipeline that bridges academic experimentation with real-world deployment architecture.

## 🔄 Multi-Source Data Engineering Architecture

The project normalizes and aligns data from disparate sources:

- **Historical & Market Data**: OHLCV and adjusted close data from **Yahoo Finance** for a multi-asset portfolio: `MSFT`, `AMZN`, `GOOGL`, `GC=F` (Gold), and `BTC-USD`.
- **Technical Features**: SMA, EMA, MACD, RSI, Bollinger Bands, momentum, and volatility.
- **Fundamental Indicators**: P/E ratio, Market Cap, and revenue growth.
- **Macroeconomic Variables**: Interest rates and inflation sourced from **FRED**.
- **Sentiment Analysis (NLP)**: Integration of a natural language processing component using **ProsusAI/finbert** (via Hugging Face) to analyze financial text for each asset.

**Project Status**: Implementation Phase (Modular Pipeline & Dashboard).

## 🧠 Modeling & Prediction Strategy

- **Classification**: Predicting upward or downward market movement (Binary Target).
- **Models**: Ensemble methods (**Random Forest**, **XGBoost**).
- **Evaluation**: Accuracy, Precision, Recall, and AUC-ROC.

## 🚀 Deployment & MLOps (Docker)

The project is containerized for portability and ease of deployment:

- **Docker**: Orchestrated environment for workers and UI.
- **Streamlit Dashboard**: Interactive UI for price, sentiment, and prediction visualization.
- **Modular Architecture**: Clear separation of ingestion, feature engineering, and modeling logic.

## 💼 Professional Summary

Designed a multi-source financial data engineering pipeline integrating market, macroeconomic, and sentiment data, with a Docker-based architecture for scalable binary classification and Streamlit visualization.

## 🛠️ Getting Started

### 1. Environment Setup
Create a `.env.local` file in the root directory:

```bash
FINNHUB_API_KEY=your_finnhub_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

### 2. Prerequisites & Requirements
The system requires the following libraries for local execution (automatically handled in Docker):

```bash
pip install yfinance pandas pandas-datareader numpy finnhub-python requests python-dotenv plotly transformers torch xgboost scikit-learn joblib streamlit
```

## 🚀 How to Launch

### Via Docker (Recommended)
This launches the **Streamlit Dashboard** and the **Worker** service in a single environment:

```bash
docker-compose up --build
```
Access the dashboard at: `http://localhost:8501`

### Local Execution (Manual)
If running outside of Docker, follow these steps in order:

1. **Quick Data Update & Initialization**:
   ```bash
   $env:PYTHONPATH="."; python -m src.data.preprocessing
   ```

2. **Model Training**:
   ```bash
   $env:PYTHONPATH="."; python -m src.models.train
   ```

3. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

## 🔗 Resources
- **Sentiment Model**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on Hugging Face.
- **Data Sources**: Yahoo Finance, FRED, Finnhub.