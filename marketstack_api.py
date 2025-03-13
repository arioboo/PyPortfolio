#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import requests

from params import access_key, endpoint, params

#< API Functions >#
def get_Heod(params=params): # - Historical Data // Free
    response = requests.get(f'{endpoint}/eod', params)
    df = pd.DataFrame(response.json()['data'])
    return df.set_index(pd.to_datetime(df.date).dt.date).sort_index(ascending=True)
    
def get_Hintraday(params=params): # - Historical Data // Free
    response = requests.get(f'{endpoint}/intraday', params)
    df = pd.DataFrame(response.json()['data'])
    return df.set_index(df.date).sort_index(ascending=True)
    
#--
def get_splits(params=params): # - Splits // Free
    response = requests.get(f'{endpoint}/splits', params)
    return pd.DataFrame(response.json()['data'])
#--
def get_dividends(params=params): # - Dividends // Free
    response = requests.get(f'{endpoint}/dividends', params)
    return pd.DataFrame(response.json()['data'])

#--
def get_ticker(t, params=params): # - Ticker // Free
    response = requests.get(f'{endpoint}/tickers/{t}', params)
    return pd.DataFrame(response.json())
    
def get_tickerlist(string, params=params): # - Tickers List // Free
    response = requests.get(f'{endpoint}/tickerslist', params | {'search': string})
    return pd.DataFrame(response.json()['data'])

def get_tickerinfo(params=params): # - Tickers Info // Free
    response = requests.get(f'{endpoint}/tickerinfo', params=params ) 
    return pd.DataFrame(response.json()['data'])

#--
def get_benchmarks(params=params): # - Stock Market Index Listing // Basic
    response = requests.get(f'{endpoint}/indexlist', params)
    return pd.DataFrame(response.json()['data'])

def get_indexinfo(index, params=params): # - Stock Market Index Info // Basic
    response = requests.get(f'{endpoint}/indexinfo', params | {'index': index})
    return pd.DataFrame(response.json())
#--
def get_exchanges(string, params=params): # - Exchanges // Free
    response = requests.get(f'{endpoint}/exchanges', params | {'search' : string})
    return pd.DataFrame(response.json()['data'])
#--
def get_currencies(params=params): # - Currencies // Free
    response = requests.get(f'{endpoint}/currencies', params)
    return pd.DataFrame(response.json()['data'])

#--
def get_timezones(params=params): # - Timezones // Free
    response = requests.get(f'{endpoint}/timezones', params)
    return pd.DataFrame(response.json()['data'])

#--
def get_bondlist(params=params): # - Bonds Listing // Basic
    response = requests.get(f'{endpoint}/bondlist', params)
    return pd.DataFrame(response.json()['data'])

def get_bondinfo(country, params=params): # - Bond Info // Basic
    response = requests.get(f'{endpoint}/bond', params | {'country' : country} )
    return pd.DataFrame(response.json()['data'])

#--
def get_etflist(ticker, params=params): # - ETF Holdings Listing // Basic
    response = requests.get(f'{endpoint}/etflist', params | {'list' : ticker})
    return pd.DataFrame(response.json()['data'])

def get_etfholdings(ticker, params=params): # - ETF Holdings Info // Basic
    response = requests.get(f'{endpoint}/etfholdings', params | {'ticker' : ticker} )
    return pd.DataFrame(response.json()['data'])

# (only missing Real-Time Data in 'Professional Plan')



def retrieve_benchmarks():

    benchs = pd.DataFrame()
    list_benchmark = get_benchmarks()
    for i in list_benchmark['benchmark']:
        benchs = pd.concat([benchs, get_indexinfo(i)])

    benchs = benchs[benchs['benchmark'].notna()].reset_index()
    benchs[benchs.filter(like='perc').columns] = \
            benchs.filter(like='perc').apply(lambda s: s.str.replace('%','').astype(float)/100)
    return benchs

#get_Heod()
#get_Hintraday()
#get_splits()
#get_dividends()
#get_ticker('MSFT')
#get_tickerlist()
#get_tickerinfo()
#get_benchmarks(params | {'limit' : 200})
#get_indexinfo('nifty_50')
#get_exchanges('Madrid')
#get_currencies()
#get_timezones()
#get_bondlist()
#get_bondinfo('Spain')
#get_etflist('SPY')
#get_etfholdings()







