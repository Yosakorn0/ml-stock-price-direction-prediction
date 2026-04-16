---
title: Machine Learning-Based Stock Prediction Using Sentiment Analysis
emoji: 🔮
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: dashboard/app.py
pinned: false
license: apache-2.0
---

# Cloud-Native Multi-Source Financial Intelligence System

**Author**: Yosakorn Sirisoot
**Academic Context**:

- **AT82.03**: Machine Learning
- **AT82.9002**: Selected Topic: Data Engineering and MLOps at AIT

## 🌟 Project Overview

This project develops an end-to-end machine learning pipeline to predict the direction of financial markets (Stocks, Forex, Gold) by integrating heterogeneous data sources: historical price data, technical indicators, fundamental indicators, macroeconomic variables, and sentiment information derived from news and social media.

The system builds a robust financial data engineering pipeline that bridges academic experimentation with real-world deployment architecture.

> [!TIP]
> For a detailed explanation of the logic, financial math, and sentiment integration, see [THEORY.md](./THEORY.md).

.
├── dashboard/                # Streamlit dashboard interface
│   └── app.py                # Main entry point for the real-time UI
│
├── src/
│   ├── data/                 # Data gathering and cleaning modules
│   │   ├── ingestion.py      # Multi-source fetching (Yahoo Finance, FRED)
│   │   ├── sentiment.py      # FinBERT sentiment analysis stage
│   │   └── preprocessing.py  # Automated feature merging and alignment
│   │
│   ├── features/             # Quantitative indicator logic
│   │   └── technical.py      # RSI, MACD, Bollinger calculations
│   │
│   └── models/               # Prediction and training logic
│       ├── train.py          # XGBoost training & evaluation pipeline
│       └── predict.py        # Real-time inference engine
│
├── infrastructure/           # Deployment and environment files
│   ├── Dockerfile            # Container definition
│   └── docker-compose.yml    # Service orchestration
│
├── data/                     # Raw and processed datasets
├── models/                   # Saved .joblib model artifacts
└── README.md


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
- **Models**: Ensemble methods (**XGBoost** with Randomized Search tuning).
- **Interpretability**: **SHAP (Shapley Additive Explanations)** for feature importance analysis.
- **Evaluation**: Accuracy, Precision, Recall, and **Annualized Sharpe Ratio** (including 0.1% transaction cost).

## 📊 Model Performance Summary

The system evaluates directional accuracy across multiple asset classes using time-series walk-forward validation.

| Asset Symbol | Baseline Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **MSFT** (Microsoft) | 0.54 -- 0.58 | 0.52 | 0.61 | 0.56 |
| **AMZN** (Amazon) | 0.52 -- 0.56 | 0.51 | 0.63 | 0.56 |
| **GOOGL** (Google) | 0.53 -- 0.57 | 0.53 | 0.59 | 0.55 |
| **GC=F** (Gold) | 0.50 -- 0.54 | 0.51 | 0.55 | 0.53 |
| **BTC-USD** (Bitcoin) | 0.51 -- 0.55 | 0.52 | 0.65 | 0.57 |

> [!NOTE]
> Performance reflects next-day binary classification (Price Up/Down). Sentiment integration from FinBERT typically provides a 2-3% marginal gain over technical-only models.

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

# This will fetch data, analyze sentiment, and train XGBoost models for all 5 assets
```bash
docker-compose run worker python -m src.data.preprocessing
docker-compose run worker python -m src.models.train
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

4. **Analyze the model**:
   ```bash
   python scripts/analyze_scientific.py
   ```

5. **Generate the report**:
   ```bash
   python scripts/generate_report.py
   ```

## 🔗 Resources
- **Sentiment Model**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) on Hugging Face.
- **Data Sources**: Yahoo Finance, FRED, Finnhub.