# Multi-Source Financial Data Architecture & ML Preparation

**Author**: Yosakorn Sirisoot
**Project Plan for**: 
- **AT82.03**: Machine Learning
- **AT82.9002**: Selected Topic: Data Engineering and MLOps at AIT

This project implements a comprehensive financial data engineering pipeline designed to support future machine learning models for asset price direction prediction (Stocks, Forex, Gold).

## 🚀 Resume-Ready Summary
Designed a multi-source financial data engineering pipeline integrating market, macroeconomic, and external validation APIs, with a planned AWS-based MLOps deployment architecture for scalable binary classification modeling.

## 🏗 Development & Tooling Environment
This project was developed using an AI-assisted coding workflow to accelerate experimentation and architectural design:
- **Cursor IDE**: Used for context-aware code generation and productivity enhancement.
- **Anti-Gravity (Google-based Tooling)**: Supported structured iteration, rapid prototyping, and validation of financial APIs during the exploratory phase.

## 🔄 Multi-Source Data Engineering Architecture
The project integrates heterogeneous data sources:
- **Market Price Data**: Yahoo Finance, Stooq (Stocks, Forex, Commodities).
- **Macroeconomic Indicators**: FRED (Federal Funds Rate).
- **Real-Time Validation**: Finnhub (Quotes & News).
- **Technical Indicator Verification**: Alpha Vantage.

The pipeline normalizes, aligns, and validates these datasets, handling varying frequencies (daily/monthly), rate limits, and disparate data schemas.

- **Project Phase**: Currently in **Data Exploration & API Evaluation**.
- **Rate Limit Resilience**: Graceful handling of Alpha Vantage API limits (5 calls/min).
- **Indicator Benchmarks**: Visual cross-verification using Plotly.

## 🎯 Two-Class Modeling Design
The system is designed for binary classification:
- **Class 1**: Next-day price increase.
- **Class 0**: Next-day price decrease.

By combining market data + macroeconomic signals + external validation, the design ensures robust feature sets for models like Random Forest or Gradient Boosting.

## ☁️ Planned AWS MLOps Integration (Roadmap)
The next phase will transition the pipeline into a production-grade AWS architecture:
- **Amazon S3**: For raw and processed data lake storage.
- **Amazon SageMaker**: For scalable model training, evaluation, and artifact tracking.
- **AWS Glue/Lambda**: For automated data ingestion and transformation layers.
- **CI/CD**: Scheduled retraining and deployment workflows.

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

## How to Use
1. Set up your `.env.local` file.
2. Open `index_1.ipynb`.
3. Run the "Dependency Setup" cell.
4. Execute exploration cells for Stocks, Forex, or Gold.
