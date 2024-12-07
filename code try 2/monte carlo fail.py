import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to get stock data and calculate daily close price percentage
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    close_prices = stock_data['Close']
    start_price = close_prices.iloc[0]
    close_prices_percentage = (close_prices / start_price) * 100
    return close_prices_percentage

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

# Function to plot Monte Carlo simulation
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

# Example usage
if __name__ == "__main__":
    ticker = "PFE"
    start_date = "2000-01-01"
    end_date = "2025-01-01"
    required_return = 50 # Required return of 120%

    # Get stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Monte Carlo simulation for the stock
    num_simulations = 100000
    forecast_period = 404  # 1 year (252 trading days)
    simulations = monte_carlo_simulation_gbm(stock_data, num_simulations, forecast_period)

    # Plot the average of Monte Carlo simulations
    plot_monte_carlo_average(stock_data, simulations, label=ticker)

    # Calculate and print the probability of success
    success_probability = calculate_probability_of_success(simulations, required_return)
    print(f"Probability of {ticker} achieving {required_return}% return: {success_probability:.2f}%")
