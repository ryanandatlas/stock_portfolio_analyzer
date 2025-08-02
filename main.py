import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.data_fetcher import fetch_stock_data
from src.portfolio_analyzer import monte_carlo_simulation, calculate_sharpe_ratio, calculate_var
from src.ml_model import predict_returns_linear, predict_returns_lstm
from src.utils import calculate_portfolio_metrics, create_return_distribution_plot, create_projection_plot, create_potential_value_table

# Streamlit app configuration
st.set_page_config(page_title="Stock Portfolio Analyzer", layout="wide")

def main():
    st.title("Stock Portfolio Analyzer")
    st.markdown("Enter your stock portfolio details to analyze potential returns, risk, and volatility.")

    # User input for portfolio
    st.subheader("Portfolio Input")
    with st.form("portfolio_form"):
        tickers_input = st.text_input("Enter stock tickers (comma-separated, e.g., AAPL,MSFT,GOOGL):")
        shares_input = st.text_input("Enter number of shares for each ticker (comma-separated):")
        investment_horizon = st.slider("Investment Horizon (years)", 1, 10, 5)
        num_simulations = st.slider("Number of Monte Carlo Simulations", 100, 1000, 500)
        submitted = st.form_submit_button("Analyze Portfolio")

    if submitted and tickers_input and shares_input:
        # Process user input
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        try:
            shares = [int(s.strip()) for s in shares_input.split(",")]
            if len(tickers) != len(shares):
                st.error("Number of tickers must match number of shares.")
                return
        except ValueError:
            st.error("Shares must be valid integers.")
            return

        # Fetch stock data
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        try:
            stock_data = fetch_stock_data(tickers, start_date, end_date)
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
            return

        # Calculate portfolio metrics
        portfolio = pd.DataFrame({"Ticker": tickers, "Shares": shares})
        weights = np.array(shares) / sum(shares)
        returns = stock_data.pct_change().dropna()
        portfolio_metrics = calculate_portfolio_metrics(returns, weights)

        # ML-based return predictions
        predicted_returns_linear = predict_returns_linear(stock_data)
        predicted_returns_lstm = predict_returns_lstm(stock_data)

        # Monte Carlo simulation
        simulations = monte_carlo_simulation(returns, weights, investment_horizon, num_simulations)
        mean_projection = np.mean(simulations, axis=1)
        lower_bound = np.percentile(simulations, 5, axis=1)
        upper_bound = np.percentile(simulations, 95, axis=1)

        # Calculate financial metrics
        sharpe_ratio = calculate_sharpe_ratio(returns, weights)
        var_95 = calculate_var(simulations[-1])

        # Calculate potential portfolio value
        initial_value = (stock_data.iloc[-1] * shares).sum()
        monte_carlo_value = np.mean(simulations[-1])
        linear_value = initial_value * (1 + predicted_returns_linear.mean()) ** investment_horizon
        lstm_value = initial_value * (1 + predicted_returns_lstm.mean()) ** investment_horizon

        # Display results
        st.subheader("Portfolio Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Expected Annual Return", f"{portfolio_metrics['expected_return']*100:.2f}%")
            st.metric("Volatility", f"{portfolio_metrics['volatility']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col2:
            st.metric("Value at Risk (95%)", f"${var_95:,.2f}")
            st.metric("Predicted Annual Return (Linear)", f"{predicted_returns_linear.mean()*100:.2f}%")
            st.metric("Predicted Annual Return (LSTM)", f"{predicted_returns_lstm.mean()*100:.2f}%")

        # Potential value section
        st.subheader("Potential Portfolio Value")
        st.plotly_chart(create_potential_value_table(initial_value, monte_carlo_value, linear_value, lstm_value, investment_horizon))

        # Visualizations
        st.subheader("Visualizations")
        st.plotly_chart(create_projection_plot(mean_projection, lower_bound, upper_bound, investment_horizon))
        st.plotly_chart(create_return_distribution_plot(simulations[-1]))

if __name__ == "__main__":
    main()