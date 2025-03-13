import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_EfficientFrontier(mu, S,
                           max_return = 0.2):
    
    from pypfopt import EfficientFrontier
    
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    performance = ef.portfolio_performance()

    # Obtener puntos de la frontera eficiente
    efrontier_returns = np.linspace(0.02, max_return, 
                                    40) # (I!)
    efrontier_risks = []

    for r in efrontier_returns:
        ef_new = EfficientFrontier(mu, S) # inicializar - instancia Optimizador
        try:
            ef_new.efficient_return(r)
            efrontier_risks.append(ef_new.portfolio_performance()[1])
        except Exception as e:
            continue

    # PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(efrontier_risks, efrontier_returns[:len(efrontier_risks)], label="Efficient Frontier")
    plt.title('Eficient Frontier')
    plt.xlabel('Risk (sigma)')
    plt.ylabel('Expected return')
    plt.scatter(performance[1], performance[0], color='red', marker='*', s=200, label="Optimal Portfolio")
    plt.legend()

def plot_returns(returns):
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, kde=True, bins=30)
    plt.title('Daily Return distribution')
    plt.show()

def corr_heatmap(returns):
    """
    Muestra un heatmap de correlación entre activos.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()