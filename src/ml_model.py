import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def predict_returns_linear(stock_data):
    returns = stock_data.pct_change().dropna()
    X = np.arange(len(returns)).reshape(-1, 1)
    model = LinearRegression()
    predictions = []
    for ticker in stock_data.columns:
        y = returns[ticker].values
        model.fit(X, y)
        future_return = model.predict([[len(returns)]])[0]
        predictions.append(future_return * 252)  # Annualize
    return pd.Series(predictions, index=stock_data.columns)

def predict_returns_lstm(stock_data, sequence_length=60):
    returns = stock_data.pct_change().dropna()
    scaler = MinMaxScaler()
    predictions = []
    
    for ticker in stock_data.columns:
        # Prepare data for LSTM
        data = returns[ticker].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length])
        X, y = np.array(X), np.array(y)
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        # Predict next return
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        predicted_return = model.predict(last_sequence, verbose=0)
        predicted_return = scaler.inverse_transform(predicted_return)[0, 0]
        predictions.append(predicted_return * 252)  # Annualize
    
    return pd.Series(predictions, index=stock_data.columns)