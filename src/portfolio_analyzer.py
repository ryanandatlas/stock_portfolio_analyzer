import numpy as np
import pandas as pd

def monte_carlo_simulation(returns, weights, years, num_simulations):
    mean_returns = returns.mean() * 252  # Annualize daily returns
    cov_matrix = returns.cov() * 252
    simulations = np.zeros((years * 252, num_simulations))
    portfolio_value = 10000  # Initial investment
    for t in range(years * 252):
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
        portfolio_return = np.sum(random_returns * weights)
        simulations[t] = portfolio_value * (1 + portfolio_return / 252)
        portfolio_value = simulations[t]
    return simulations

def calculate_sharpe_ratio(returns, weights, risk_free_rate=0.01):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return (portfolio_return - risk_free_rate) / portfolio_volatility

def calculate_var(portfolio_values, confidence_level=0.95):
    return np.percentile(portfolio_values, (1 - confidence_level) * 100)