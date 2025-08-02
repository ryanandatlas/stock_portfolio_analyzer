import yfinance as yf
import pandas as pd
import time

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data for given tickers.
    
    Args:
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: DataFrame with closing prices for each ticker.
    """
    try:
        time.sleep(1)  # Avoid rate limiting
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False, threads=False)
        if data.empty or data.isna().all().all():
            raise ValueError(f"No data returned for tickers: {tickers}")
        
        # Handle single ticker
        if len(tickers) == 1:
            if 'Adj Close' in data.columns:
                # Single ticker may return a Series or DataFrame
                if isinstance(data['Adj Close'], pd.Series):
                    return data['Adj Close'].to_frame(name=tickers[0])
                else:
                    return data[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
            elif 'Close' in data.columns:
                if isinstance(data['Close'], pd.Series):
                    return data['Close'].to_frame(name=tickers[0])
                else:
                    return data[['Close']].rename(columns={'Close': tickers[0]})
            else:
                raise ValueError(f"No 'Adj Close' or 'Close' column for {tickers}")
        # Handle multiple tickers
        else:
            if 'Adj Close' in data.columns:
                return data['Adj Close']
            elif 'Close' in data.columns:
                return data['Close']
            else:
                raise ValueError(f"No 'Adj Close' or 'Close' columns for {tickers}")
    except Exception as e:
        raise Exception(f"Failed to fetch stock data for {tickers}: {str(e)}")