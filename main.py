
from pylab import plt
plt.style.use('seaborn-v0_8')
import warnings
from arch.univariate.base import DataScaleWarning, ConvergenceWarning
warnings.simplefilter("ignore", DataScaleWarning)
warnings.simplefilter("ignore", ConvergenceWarning)

from pypfopt import expected_returns, risk_models
from pypfopt import BlackLittermanModel #- bayesian-based
from pypfopt import EfficientFrontier, EfficientCVaR, HRPOpt, CLA

from params import *

import marketstack_api as api
from analysis import *
from bayesian import update_bayesian_posterior
from plotting import plot_EfficientFrontier, plot_returns

if __name__ == "__main__":

    # -- < returns - Benchmark Index > --
    list_benchmark = api.get_benchmarks()

    us500 = api.get_Heod(params | {'symbols':'US500.INDX'}) 
    us500 = us500.pivot(columns='symbol', values=value_choice).dropna() 

    market_returns = compute_returns(us500, log_returns=make_log_returns)
   

    # -- < returns - Assets > --
    params.update(
        {'date_from' : date_from, 
         'date_to' : date_to, 
         'limit':10000} )

    data = api.get_Heod(params)
    data = data.pivot(columns='symbol', values=value_choice).dropna()
    
    returns = compute_returns(data, log_returns=make_log_returns) 
    
    # -- < Market Betas > --
    idx_intersection = returns.index.intersection(market_returns.index)
    returns.attrs['beta'] = {} # Betas del portfolio
    for s in returns:
        returns.attrs['beta'][s] = compute_beta(asset_returns=returns.loc[idx_intersection][s], 
                                       market_returns=market_returns.loc[idx_intersection]['US500.INDX'])
        
    # -- < Stats and Tests > --
    data.attrs['max_drawdown'], us500.attrs['max_drawdown'] =\
          [max_drawdown(df) for df in [data,us500]]
    #returns.skew(), returns.kurtosis())
    returns.attrs['VaR'], returns.attrs['cVaR'] = var_cvar(returns)

    tests = {}
    tests = normality_tests(returns)
    
    # expected returns and covariance matrix
    if not make_posterior_returns:
        mu = expected_returns.mean_historical_return(prices=data)
    else:
        print('< WITH BAYESIAN PRIOR >')
        bl = update_bayesian_posterior(Q=Q, P=P, data=data,
                                       tau=tau)
        mu = bl.bl_returns()
    S = risk_models.sample_cov(prices=data) 
        
    # -- Models --
    garch = {}
    garch['us500'] = fit_garch(market_returns, 1,1)
    for serie in returns:
        garch[serie] = fit_garch(returns[serie], 1,1)

    # -- Optimization --
    models = analysis_all_models(mu=mu, S=S, 
                                 returns=returns,
                                 weight_constraints=weight_constraints) # main function
   

    print (f'Model_Name:{model_name} | Optimization:{optimization}\n')
    choosen_model = models.loc[(models['model_name']==model_name) & 
                               (models['optimization']==optimization)
                            ]
    print("Performance:", 
          'exp_return=%.3f'%choosen_model['return'].iloc[0], 
          'volatility=%.3f'%choosen_model['risk'].iloc[0], 
          'sharpe=%.3f'%choosen_model['sharpe'].iloc[0], '\n')
    print("Optimal weights:")
    display(choosen_model['weights'].iloc[0])

    # -- Plotting --
    if make_plots:
        plot_EfficientFrontier(mu,S)
        plt.savefig('imgs/EfficientFrontier.png') if save_plots else None

        plot_returns(returns)
        plt.savefig('imgs/Returns_dist.png') if save_plots else None

        plot_returns(market_returns)
        plt.savefig('imgs/mktReturns_dist.png') if save_plots else None

    
    
