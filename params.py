from numpy import array, zeros_like
from pandas import get_dummies
from json import load

with open('config.json','r') as f:
    config = load(f)
    for v in config.values():
        globals().update(v)
    
#-- API ~ ./marketstack_api.py
from io_aux import read_csv
inputs = read_csv(inputs_file)
params = {
    'access_key' : access_key,
    'symbols' : ','.join(inputs.ticker),
    } # otros: date_from, date_to, symbol(s), ticker(s), 
date_from, date_to = date_range

#- main.py

weight_constraints = [lambda w: w >= w_lim[0],
                      lambda w: w <= w_lim[-1]
                      ]     # 1%<w<60%

#(opt) < Bayesian > --
expectation_vector = zeros_like(inputs.weight, dtype=float)
Q = array(Q)        # =returns expectatives (for tickers)
P = array(P)        # =asset relations to Q (-1 to 1)







'''
NOTES:
Toda la configuración esta en 'config.json'. Este archivo preprocesa estos datos para posteriores scripts.

model_name: 'EfficientFrontier', 'EfficientCVaR', 'CLA', 'HRPOpt'
optimization: 'min_volatility', 'max_sharpe', 'efficient_return',
       'efficient_risk', 'max_quadratic_utility', 'min_cvar', '-'


- https://marketstack.com/documentation_v2
'''
