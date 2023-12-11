#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:46:45 2023

@author: lucasmuyor
"""

import yfinance as yf
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Define the tickers for the three financial institutions
tickers = ['JPM', 'BAC', 'WFC']

# Define the time period for data extraction
start_date = '2012-01-01'
end_date = '2022-12-31'


# Download daily price data for each ticker
price_data = {ticker: yf.download(ticker, start=start_date, end=end_date)['Close'] for ticker in tickers}

# Convert the closing prices to returns
returns_data = {ticker: np.log(price_data[ticker]).diff().dropna() for ticker in tickers}

# Focus analysis on the period 2012-2018
analysis_end_date = '2018-12-31'

# Analyze each time series
for ticker in returns_data:
    returns = returns_data[ticker].loc[:analysis_end_date]

    # Perform Augmented Dickey-Fuller test for stationarity
    adf_result = adfuller(returns)
    print(f'{ticker} ADF Statistic: {adf_result[0]}')
    print(f'{ticker} p-value: {adf_result[1]}')

    # Plotting returns to visualize trends and volatility
    plt.figure(figsize=(10, 4))
    plt.plot(returns)
    plt.title(f'{ticker} Daily Returns (2012-2018)')
    plt.ylabel('Returns')
    plt.xlabel('Date')
    plt.show()

    # Fit a GARCH(1,1) model to the returns
    model = arch_model(returns, vol='Garch', p=1, q=1)
    results = model.fit()
    print(results.summary())
    
forecast_start_date = '2019-01-01'

forecasted_variance = {}

for ticker in returns_data:
    returns = returns_data[ticker].loc[:analysis_end_date]
    
    model = arch_model(returns, vol='Garch', p=1, q=1)
    results = model.fit()

    # Utiliser le dernier modèle pour faire des prévisions pour la période 2019-2022
    forecasts = results.forecast(start=forecast_start_date, horizon=1)
    forecasted_variance[ticker] = forecasts.variance[forecast_start_date:end_date]['h.1']
    print(f"Fitting results for {ticker}:\n", results.summary(), "\n")
    
    try:
        forecasts = results.forecast(start=forecast_start_date, horizon=1)
        forecasted_variance[ticker] = forecasts.variance[forecast_start_date:]['h.1']
        print(f"{ticker} forecasted variance head:\n", forecasted_variance[ticker].head(), "\n")
    except Exception as e:
        print(f"Error in forecasting for {ticker}: {e}")
    
for ticker, variance in forecasted_variance.items():
    plt.figure(figsize=(10, 4))
    plt.plot(variance, label=f'{ticker} Forecasted Variance')
    plt.title(f'Forecasted Conditional Variance for {ticker} (2019-2022)')
    plt.ylabel('Forecasted Variance')
    plt.xlabel('Date')
    plt.legend()
    plt.show()

