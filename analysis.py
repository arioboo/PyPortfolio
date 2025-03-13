import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import minimize
from arch import arch_model

# 1. Cálculo de retornos y estadísticas básicas
def compute_returns(prices: pd.Series, log_returns=False):
    """Calcula los retornos simples o logarítmicos de una serie de precios."""
    if log_returns:
        return prices.pct_change().dropna() 
    else:
        return np.log(prices/prices.shift(1)).dropna()

# Media y volatilidad
def compute_stats(returns: pd.Series):
    """Calcula la media y la volatilidad de los retornos."""
    mean_return = returns.mean()
    volatility = returns.std()
    return mean_return, volatility

# Coeficiente de variación
def coefficient_of_variation(returns: pd.Series):
    return returns.std() / returns.mean()

# 2. Forma de la distribución
def compute_skewness_kurtosis(returns: pd.Series):
    """Calcula la asimetría y curtosis de los retornos."""
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    return skewness, kurtosis

# Prueba de normalidad
def normality_tests(returns: pd.Series):
    """Realiza pruebas de normalidad (Jarque-Bera y Shapiro-Wilk)."""
    jb_test = stats.jarque_bera(returns)
    shapiro_test = stats.shapiro(returns)
    return {"Jarque-Bera": jb_test, "Shapiro-Wilk": shapiro_test}

# 3. Riesgo: VaR y CVaR
def var_cvar(returns: pd.Series, alpha=0.05):
    """Calcula el VaR y el CVaR al nivel de confianza especificado."""
    var = np.percentile(returns, 100 * alpha)
    cvar = returns[returns <= var].mean()
    return var, cvar

# Drawdown máximo
def max_drawdown(prices: pd.Series):
    """Calcula el máximo drawdown de una serie de precios."""
    cumulative_returns = prices / prices.iloc[0]
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns - rolling_max
    max_dd = drawdown.min()
    return max_dd

# 4. Modelado y pronóstico con GARCH
def fit_garch(returns: pd.Series, p=1, q=1):
    """Ajusta un modelo GARCH(1,1) a los retornos."""
    model = arch_model(returns.dropna(), vol='Garch', p=p, q=q)
    result = model.fit(disp='off')
    return result.summary()

# 5. Beta del activo y CAPM
def compute_beta(asset_returns: pd.Series, market_returns: pd.Series):
    """Calcula la beta del activo respecto al mercado."""
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    return beta

# 6. Optimización de portafolio con Markowitz
def optimize_portfolio(returns: pd.DataFrame):
    """Optimiza la asignación de activos en un portafolio usando el modelo de Markowitz."""
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)
    
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)  # @ es mult matricial
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(portfolio_volatility, initial_weights, bounds=bounds, constraints=constraints)
    
    return result.x  # Retorna los pesos óptimos

def fit_returns(returns):
    from scipy.stats import norm
    mu, sigma = norm.fit(returns)
    print(f'mu:{mu}\nsigma:{sigma}')
    return mu, sigma

def sensibility_proof(returns, perturbation=0.01):
    ''''
    Evalúa cómo pequeños cambios en los retornos afectan la optimización.
    '''
    pert_matrix = np.random.normal(0, perturbation, returns.shape)
    pert_returns = returns + pert_matrix
    print("Simulated variation in returns:")
    print(pert_returns.describe())


def analysis_all_models(mu, S, 
                        returns,
                        weight_constraints : list):
    from pypfopt import EfficientFrontier, EfficientCVaR, HRPOpt, CLA

    df = pd.DataFrame(columns=['model_name','optimization',
                               'return','risk','sharpe',
                               'weights'])
    
    ef = EfficientFrontier(mu, S)
    [ ef.add_constraint(constraint) for constraint in weight_constraints] if weight_constraints else None
    weights = ef.min_volatility()
    df.loc[0] = [EfficientFrontier.__name__, 'min_volatility' ] \
                + list(ef.portfolio_performance()) + [dict(weights)]
    
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=S)
    [ ef.add_constraint(constraint) for constraint in weight_constraints] if weight_constraints else None
    weights = ef.max_sharpe()
    df.loc[1] = [EfficientFrontier.__name__, 'max_sharpe' ] \
                + list(ef.portfolio_performance()) + [dict(weights)]
    
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=S)
    [ ef.add_constraint(constraint) for constraint in weight_constraints] if weight_constraints else None
    weights = ef.efficient_return(target_return=0.1)
    df.loc[2] = [EfficientFrontier.__name__, 'efficient_return' ] \
                + list(ef.portfolio_performance()) + [dict(weights)]
    
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=S)
    [ ef.add_constraint(constraint) for constraint in weight_constraints] if weight_constraints else None
    weights = ef.efficient_risk(target_volatility=0.3)
    df.loc[3] = [EfficientFrontier.__name__, 'efficient_risk' ] \
                + list(ef.portfolio_performance()) + [dict(weights)]
    
    ef = EfficientFrontier(expected_returns=mu, cov_matrix=S)
    [ ef.add_constraint(constraint) for constraint in weight_constraints] if weight_constraints else None
    weights = ef.max_quadratic_utility()
    df.loc[4] = [EfficientFrontier.__name__, 'max_quadratic_utility' ] \
                + list(ef.portfolio_performance()) + [dict(weights)]
    
    evar = EfficientCVaR(expected_returns=mu, returns=returns)
    [ evar.add_constraint(constraint) for constraint in weight_constraints] if weight_constraints else None
    weights = evar.min_cvar()
    df.loc[5] = [EfficientCVaR.__name__, 'min_cvar' ] \
                + list(evar.portfolio_performance()) + [np.nan] + [dict(weights)]
    
    cla = CLA(expected_returns=mu, cov_matrix=S)
    weights = cla.min_volatility()
    df.loc[6] = [CLA.__name__, 'min_volatility'] \
                + list(cla.portfolio_performance()) + [dict(weights)]
    
    cla = CLA(expected_returns=mu, cov_matrix=S)
    weights = cla.max_sharpe()
    df.loc[7] = [CLA.__name__, 'max_sharpe'] \
                + list(cla.portfolio_performance()) + [dict(weights)]
    
    hrp = HRPOpt(returns=returns)
    weights = hrp.optimize()
    df.loc[8] = [HRPOpt.__name__, '-'] \
                + list(hrp.portfolio_performance()) + [dict(weights)]
    
    #da = DiscreteAllocation()

    return df

