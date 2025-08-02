import yfinance as yf
import pandas as pd
tickers = ['AAPL']
start_date = '2020-01-01'
end_date = '2025-08-02'
try:
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False, threads=False)
    print("Columns:", data.columns)
    print("Data head:\n", data.head())
    if len(tickers) == 1:
        if 'Adj Close' in data.columns:
            if isinstance(data['Adj Close'], pd.Series):
                print("Single ticker, Series:\n", data['Adj Close'].to_frame(name=tickers[0]).head())
            else:
                print("Single ticker, DataFrame:\n", data[['Adj Close']].rename(columns={'Adj Close': tickers[0]}).head())
        else:
            print("Single ticker, Close:\n", data[['Close']].rename(columns={'Close': tickers[0]}).head())
    else:
        print("Multi-ticker Adj Close:\n", data['Adj Close'].head())
except Exception as e:
    print(f"Error: {e}")