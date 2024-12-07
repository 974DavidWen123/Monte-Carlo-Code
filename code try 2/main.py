import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Function to get stock data and calculate daily close price percentage
def get_stock_data(ticker, start_date, end_date, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                close_prices = stock_data['Close']
                start_price = close_prices.iloc[0]
                close_prices_percentage = (close_prices / start_price) * 100
                return close_prices_percentage
            else:
                print(f"No data available for {ticker}")
                return None
        except Exception as e:
            retries += 1
            print(f"Failed to download {ticker}: {e}. Retrying {retries}/{max_retries}...")
            time.sleep(2)  # Wait before retrying

    print(f"Failed to download data for {ticker} after {max_retries} attempts.")
    return None

# LSTM Model for predicting returns
def lstm_model(returns, epochs=10, batch_size=32):
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

    predicted_returns_df = pd.DataFrame(predicted_returns, columns=returns.columns)
    return predicted_returns_df

# Function for Monte Carlo simulation with geometric Brownian motion
def monte_carlo_simulation_gbm(data, num_simulations, forecast_period, days=252):
    returns = data.pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()

    simulations = np.zeros((forecast_period, num_simulations))
    simulations[0, :] = data.iloc[-1]

    for day in range(1, forecast_period):
        drift = (mu - 0.5 * sigma**2) * (days / 252)
        diffusion = sigma * np.sqrt(days / 252) * np.random.normal(size=num_simulations)
        simulations[day, :] = simulations[day - 1, :] * np.exp(drift + diffusion)

    return simulations

# Function to calculate probability of success
def calculate_probability_of_success(simulations, required_return):
    success = np.mean(simulations[-1, :] >= required_return) * 100
    return success

# Function to plot the average of Monte Carlo simulations
def plot_monte_carlo_average(data, simulations, label):
    plt.plot(data.index, data, label=label + " (Actual)", linewidth=2, color='blue')

    # Calculate and plot the average of simulations
    average_simulation = np.mean(simulations, axis=1)
    plt.plot(data.index[-1] + pd.to_timedelta(np.arange(1, len(average_simulation) + 1), unit='D'), 
             average_simulation, label=label + " (Average Prediction)", linestyle='--', linewidth=2, color='red')
    
    plt.xlabel("Date")
    plt.ylabel("Price Percentage")
    plt.legend()
    plt.show()

# Main function for running both LSTM and Monte Carlo simulations on a portfolio or individual stock
if __name__ == "__main__":

    use_portfolio = True  # Set to False if using a single stock

    if use_portfolio:
        # Define portfolio
        portfolio = {"JPM": 0.15, "QQQ": 0.15, "CMG": 0.05, "NVDA": 0.1, "NFLX": 0.07, "GOOG": 0.05, 
                     "XOM": 0.07, "WM": 0.05, "AMZN": 0.07, "C": 0.03, "RY": 0.05, "KO": 0.03, 
                     "BAC": 0.03, "MA": 0.05, "META": 0.05}

        start_date_portfolio = "2022-01-01"
        end_date_portfolio = "2023-10-01"

        # Fetch historical data for portfolio
        returns = pd.DataFrame()
        for stock_ticker, weight in portfolio.items():
            close_prices_percentage = get_stock_data(stock_ticker, start_date_portfolio, end_date_portfolio)
            if close_prices_percentage is not None:
                if returns.empty:
                    returns[stock_ticker] = close_prices_percentage * weight
                else:
                    returns[stock_ticker] = close_prices_percentage * weight

        # Drop NaN values
        returns = returns.dropna()

        # Predict returns with the LSTM model for portfolio
        predicted_returns_df = lstm_model(returns)
        print("Predicted Returns:")
        print(predicted_returns_df)

        # Monte Carlo simulation for portfolio
        num_simulations_portfolio = 1000
        forecast_period_portfolio = 66  # Approximately 3 months
        portfolio_simulations = monte_carlo_simulation_gbm(returns.mean(axis=1), num_simulations_portfolio, forecast_period_portfolio)

        # Plot the average of Monte Carlo simulations for portfolio
        plot_monte_carlo_average(returns.mean(axis=1), portfolio_simulations, label="Portfolio")

        # Calculate probability of success (example: required return is 105%)
        required_return = 7
        success_probability = calculate_probability_of_success(portfolio_simulations, required_return)
        print(f"Portfolio probability of success to achieve {required_return}% return: {success_probability:.2f}%")

    else:
        # Example for a single stock
        stock_ticker = "AAPL"
        start_date = "2000-01-01"
        end_date = "2023-01-01"
        required_return = 120  # Example required return: 120%

        # Get stock data
        stock_data = get_stock_data(stock_ticker, start_date, end_date)

        # Monte Carlo simulation for the stock
        num_simulations = 1000000
        forecast_period = 504  # 1 year (252 trading days)
        stock_simulations = monte_carlo_simulation_gbm(stock_data, num_simulations, forecast_period)

        # Plot the average of Monte Carlo simulations for the stock
        plot_monte_carlo_average(stock_data, stock_simulations, label=stock_ticker)

        # Calculate and print the probability of success
        success_probability = calculate_probability_of_success(stock_simulations, required_return)
        print(f"{stock_ticker} probability of achieving {required_return}% return: {success_probability:.2f}%")
