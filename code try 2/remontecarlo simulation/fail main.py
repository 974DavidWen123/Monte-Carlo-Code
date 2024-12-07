import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Function to get stock data and calculate daily close price percentage
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    close_prices = stock_data['Close']
    start_price = close_prices.iloc[0]
    close_prices_percentage = (close_prices / start_price) * 100
    return close_prices_percentage, stock_data

# Function to build and run the LSTM model
def lstm_model(returns, epochs=50, batch_size=32):
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

# Function to calculate probability of success
def calculate_probability_of_success(simulations, required_return):
    success = np.mean(simulations[-1, :] >= required_return) * 100
    return success

# Function to plot predicted returns
def plot_predicted_returns(predicted_returns_df):
    plt.figure(figsize=(10, 6))
    plt.bar(predicted_returns_df.columns, predicted_returns_df.values[0], color='blue')
    plt.axhline(y=105, color='r', linestyle='--', label='Required Return (105%)')
    plt.xlabel('Stocks')
    plt.ylabel('Predicted Returns (%)')
    plt.title('Predicted Returns of Portfolio Stocks')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define a stock portfolio
    portfolio = {"AAPL": 1.0}
    
    start_date = "2018-01-01"
    end_date = "2023-01-01"

    # Fetch historical data for portfolio
    returns = pd.DataFrame()
    for stock_ticker, weight in portfolio.items():
        close_prices_percentage, stock_data = get_stock_data(stock_ticker, start_date, end_date)
        returns[stock_ticker] = close_prices_percentage * weight

    # Drop NaN values
    returns = returns.dropna()

    # Predict returns with the LSTM model
    predicted_returns_df = lstm_model(returns)

    print("Predicted Returns:")
    print(predicted_returns_df)

    # Calculate and print probability of success
    required_return = 30  # Required return of 30%
    simulations = np.random.normal(predicted_returns_df.values.flatten(), 0.05, (1000, len(predicted_returns_df.columns)))  # Example simulations
    success_probability = calculate_probability_of_success(simulations, required_return)
    print(f"Probability of achieving {required_return}% return: {success_probability:.2f}%")

    # Plot predicted returns
    plot_predicted_returns(predicted_returns_df)
