import pandas as pd
import numpy as np
import plotly.graph_objects as go

def calculate_portfolio_metrics(returns, weights):
    expected_return = np.sum(returns.mean() * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return {"expected_return": expected_return, "volatility": volatility}

def create_projection_plot(mean_projection, lower_bound, upper_bound, years):
    fig = go.Figure()
    time = np.linspace(0, years, len(mean_projection))
    fig.add_trace(go.Scatter(x=time, y=mean_projection, mode="lines", name="Mean Projection"))
    fig.add_trace(go.Scatter(x=time, y=upper_bound, mode="lines", name="95% Upper Bound", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=time, y=lower_bound, mode="lines", name="5% Lower Bound", line=dict(dash="dash")))
    fig.update_layout(title="Portfolio Value Projection", xaxis_title="Years", yaxis_title="Portfolio Value ($)", template="plotly_dark")
    return fig

def create_return_distribution_plot(final_values):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=final_values, nbinsx=50, name="Final Portfolio Values"))
    fig.update_layout(title="Distribution of Projected Portfolio Values", xaxis_title="Portfolio Value ($)", yaxis_title="Frequency", template="plotly_dark")
    return fig

def create_potential_value_table(initial_value, monte_carlo_value, linear_value, lstm_value, years):
    data = {
        "Method": ["Initial Value", "Monte Carlo", "Linear Regression", "LSTM"],
        f"Value in {years} Years ($)": [
            f"{initial_value:,.2f}",
            f"{monte_carlo_value:,.2f}",
            f"{linear_value:,.2f}",
            f"{lstm_value:,.2f}"
        ]
    }
    df = pd.DataFrame(data)
    fig = go.Figure(data=[
        go.Table(
            header=dict(values=list(df.columns), fill_color='paleturquoise', align='left'),
            cells=dict(values=[df.Method, df[f"Value in {years} Years ($)"]], fill_color='lavender', align='left')
        )
    ])
    fig.update_layout(title="Potential Portfolio Value Comparison", template="plotly_dark")
    return fig