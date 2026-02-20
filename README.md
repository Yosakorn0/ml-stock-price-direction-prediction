# ML Stock Price Prediction

This project implements a machine learning model to predict the direction of a stock's price movement (Up or Down) for the next day. It utilizes historical stock prices and macroeconomic indicators to build a predictive binary classification model.

- **Project Phase**: Currently in **Data Exploration & API Evaluation**.
- **Multi-Source Stock Data**: Integrated with **Yahoo Finance** for historical data and **Finnhub** for real-time market validation.
- **Expert Exploration Utilities**:
    - **Rate Limit Resilience**: Graceful handling of Alpha Vantage API limits.
    - **Indicator Benchmarks**: Visual cross-verification using Plotly.
- **Data Integrity**: Automated checks for missing values and cross-source consistency validation.

## Environment Setup

To run this project securely, create a file named `.env.local` in the root directory and add your API keys:

```bash
FINNHUB_API_KEY=your_finnhub_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

## Prerequisites

To run the project, ensure you have the following Python libraries installed:

```bash
pip install yfinance pandas pandas-datareader numpy finnhub-python requests python-dotenv plotly
```

## How to Use

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```
2. **Open the Notebook**:
   Launch Jupyter Notebook or any compatible IDE (like VS Code or JupyterLab) and open `index.ipynb`.
3. **Run the Cells**:
   Follow the step-by-step implementation in the notebook cells to fetch data, engineer features, and inspect the dataset structure.

## Project Structure

- `index.ipynb`: The main notebook containing data fetching, processing, and feature engineering logic.
- `images/`: Directory containing project-related visuals.
- `README.md`: Project documentation.

## Example Output

The project includes an integrity check and data preview:

```text
--- Dataset Structure (stock_data.info) ---
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 1479 entries, 2020-03-13 to 2026-01-30
Data columns (total 10 columns):
...
--- Integrity Check (Null Values) ---
Close           0
Target          0
RSI             0
...
```

## License

This project is open-source and available for educational purposes.
