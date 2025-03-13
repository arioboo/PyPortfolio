
from params import Q,P
from pypfopt import expected_returns, risk_models

def update_bayesian_posterior(Q,P, data,
                              tau=0.05):
    from pypfopt import BlackLittermanModel
    
    S = risk_models.sample_cov(prices=data)
    pi = expected_returns.capm_return(data) #prior
    
    bl = BlackLittermanModel(cov_matrix=S, pi=pi,
                                Q=Q, P=P,
                                tau=tau)
    #mu = bl.bl_returns()  # posterior_returns
    return bl  