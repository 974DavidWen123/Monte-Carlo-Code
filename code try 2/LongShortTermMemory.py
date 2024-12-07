import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to get stock data and calculate daily close price percentage
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    close_prices = stock_data['Close']
    start_price = close_prices.iloc[0]
    close_prices_percentage = (close_prices / start_price) * 100
    return close_prices_percentage

# Function to build and run the LSTM model
def lstm_model(returns, epochs=10, batch_size=32):
    # LSTM input preparation
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(returns.values)

    # Prepare input data
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, :])
        y.append(scaled_data[i, :])

    X, y = np.array(X), np.array(y)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=len(returns.columns)))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    # Make predictions
    inputs = scaled_data[-60:, :]
    inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
    predicted_returns = model.predict(inputs)
    predicted_returns = scaler.inverse_transform(predicted_returns)

    # Display predicted returns
    predicted_returns_df = pd.DataFrame(predicted_returns, columns=returns.columns)
    return predicted_returns_df

# Example usage
if __name__ == "__main__":
    # Define a stock portfolio
    portfolio = {"GOOG": 1}
    
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    # Fetch historical data for portfolio
    returns = pd.DataFrame()
    for stock_ticker, weight in portfolio.items():
        close_prices_percentage = get_stock_data(stock_ticker, start_date, end_date)
        if returns.empty:
            returns[stock_ticker] = close_prices_percentage * weight
        else:
            returns[stock_ticker] = close_prices_percentage * weight

    # Drop NaN values
    returns = returns.dropna()

    # Predict returns with the LSTM model
    predicted_returns_df = lstm_model(returns)
    print("Predicted Returns:")
    print(predicted_returns_df)


   