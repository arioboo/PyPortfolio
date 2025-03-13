# %%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Monte Carlo Valuation of European Call Option

def monte_carlo_valuation(S0, K, T, r, sigma, I=100000):
    """
    Monte Carlo valuation of a European call option in the Black-Scholes-Merton model.

    Parameters:
    S0 : float : Initial stock price
    K : float : Strike price
    T : float : Time to maturity
    r : float : Risk-free interest rate
    sigma : float : Volatility
    I : int : Number of Monte Carlo simulations

    Returns:
    float : Estimated call option value
    """
    np.random.seed(1000)
    z = np.random.standard_normal(I)  # Generate random standard normal numbers
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
    hT = np.maximum(ST - K, 0)
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0

# Function for plotting S&P 500 index and rolling volatility
def plot_sp500_volatility(data_file):
    """
    Reads S&P 500 index data and plots its values along with rolling volatility.

    Parameters:
    data_file : str : Path to the CSV file containing S&P 500 index data
    """
    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    data = pd.DataFrame(data['.SPX'])
    data.dropna(inplace=True)

    data['rets'] = np.log(data / data.shift(1))
    data['vola'] = data['rets'].rolling(252).std() * np.sqrt(252)

    data[['.SPX', 'vola']].plot(subplots=True, figsize=(10, 6))
    plt.show()

# Function for Performance Optimization Comparisons
def compute_function_performance():
    """
    Compares execution speed of different implementations for evaluating a mathematical function.
    """
    import numexpr as ne

    loops = 2500000
    a = np.arange(1, loops)

    def f(x):
        return 3 * np.log(x) + np.cos(x) ** 2

    %timeit result = [f(x) for x in a]
    %timeit result = 3 * np.log(a) + np.cos(a) ** 2

    ne.set_num_threads(1)
    %timeit result = ne.evaluate('3 * log(a) + cos(a) ** 2')

    ne.set_num_threads(4)
    %timeit result = ne.evaluate('3 * log(a) + cos(a) ** 2')

# Example usage
if __name__ == "__main__":
    # Monte Carlo Valuation Example
    option_value = monte_carlo_valuation(S0=100, K=105, T=1.0, r=0.05, sigma=0.2)
    print(f"Estimated European Call Option Value: {option_value:.3f}")

    # Uncomment the line below to plot S&P 500 Volatility (requires data file)
    # plot_sp500_volatility('sp500_data.csv')

    # Run performance test (commented to prevent long execution in normal runs)
    # compute_function_performance()



