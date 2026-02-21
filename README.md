# Cloud-Native Multi-Source Financial Intelligence System

**Author**: Yosakorn Sirisoot
**Academic Context**: 
- **AT82.03**: Machine Learning
- **AT82.9002**: Selected Topic: Data Engineering and MLOps at AIT

## 🌟 Project Overview
This project develops an end-to-end machine learning pipeline to predict the direction of financial markets (Stocks, Forex, Gold) by integrating heterogeneous data sources: historical price data, technical indicators, fundamental indicators, macroeconomic variables, and sentiment information derived from news and social media.

The system builds a robust financial data engineering pipeline that bridges academic experimentation with real-world deployment architecture.

## 🏗 Development & Tooling Environment
- **Cursor IDE**: Enhanced productivity through context-aware code generation and refactoring.
- **Anti-Gravity (Google-based Tooling)**: Supported structured iteration, rapid prototyping, and validation of financial APIs during the exploratory phase.

## 🔄 Multi-Source Data Engineering Architecture
The project normalizes and aligns data from disparate sources:
- **Historical & Market Data**: OHLCV and adjusted close data from **Yahoo Finance** and **Stooq**.
- **Technical Features**: SMA, EMA, MACD, RSI, Bollinger Bands, momentum, and volatility.
- **Fundamental Indicators**: P/E ratio, EPS, revenue growth, and debt ratio.
- **Macroeconomic Variables**: Interest rates, inflation, GDP, oil prices, and the USD index sourced from **FRED** and **World Bank**.
- **Sentiment Analysis (NLP)**: Integration of a natural language processing component using **FinBERT** to analyze financial text from **NewsAPI**, **Reddit**, and **Twitter**.

**Project Status**: Currently in **Data Exploration & API Evaluation Phase**.

## 🧠 Modeling & Prediction Strategy
The system is designed to handle both:
- **Classification**: Predicting upward or downward market movement (Binary Target).
- **Regression**: Predicting log returns.

Evaluated models include linear models, ensemble methods (**Random Forest**, **XGBoost**), and deep learning approaches (**LSTM**), assessed via accuracy, RMSE, and Sharpe ratio.

## ☁️ Planned AWS MLOps Integration (Roadmap)
To ensure scalability, the project is designed for a cloud-native MLOps framework on **AWS**:
- **Amazon S3**: Automated data ingestion and data lake storage.
- **Amazon SageMaker**: Orchestrated preprocessing, model training, evaluation, and deployment.
- **Amazon CloudWatch**: Continuous monitoring and performance tracking.
- **CI/CD**: Automated retraining loops using updated market data.

## 💼 Professional Summary
Designed a multi-source financial data engineering pipeline integrating market, macroeconomic, and sentiment APIs, with a planned AWS-based MLOps architecture for scalable binary classification and regression modeling.

## Environment Setup
Create a `.env.local` file in the root directory:
```bash
FINNHUB_API_KEY=your_finnhub_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

## Prerequisites
```bash
pip install yfinance pandas pandas-datareader numpy finnhub-python requests python-dotenv plotly
```
