# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 15:23:47 2021

@author: Wu
"""

import pandas as pd
from scipy.stats import shapiro
from scipy.stats import normaltest
#import calplot
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualization(comp):
    plt.title('Dist of daily returns')
    comp['Portfolio'].plot.hist(bins=30)
    
    plt.figure()
    
    comp.plot.scatter(x='Portfolio', y='Benchmark')
    plt.title('Port ret vs. benchmark ret')
    plt.figure()
    comp.plot.hexbin(x='Portfolio', y='Benchmark', gridsize=12, reduce_C_function=np.mean)
    
    print('Mean, skew, kurtosis: ', comp['Portfolio'].mean(), comp['Portfolio'].skew(), comp['Portfolio'].kurtosis())
    
    stat, p = shapiro(comp['Portfolio'].dropna())
    print('Shapiro-Wilk Test: Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Sample looks Gaussian (fail to reject H0)')
    else:
    	print('Sample does not look Gaussian (reject H0)')
    
    
    stat, p = normaltest(comp['Portfolio'].dropna())
    print('D’Agostino’s K^2 Test: Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
    	print('Sample looks Gaussian (fail to reject H0)')
    else:
    	print('Sample does not look Gaussian (reject H0)')
    
    print('Annualized return: ', (comp['Portfolio']+1).prod()**(252/len(comp))-1)
    print('Sharpe ratio: ', np.sqrt(250)*np.nanmean(comp['Portfolio'] - comp['Benchmark'])/np.nanstd(comp['Portfolio'] - comp['Benchmark']))
    print('Sortino ratio: ', np.sqrt(250)*np.nanmean(comp['Portfolio'] - 0)/np.nanstd(comp['Portfolio'][comp['Portfolio'] < 0]))
    
    plt.figure()
    cum_ret=(comp+1).cumprod() -1
    print('Max DD: ', cum_ret['Portfolio'][cum_ret['Portfolio'].idxmax():].min()/cum_ret['Portfolio'].max() - 1)
    print(cum_ret.corr(method='pearson'))
    
    cum_ret.plot()
    plt.title('Cum ret')
    # Plot annual returns
    plt.figure()
    
    annual_returns = comp + 1
    annual_returns['Year'] = pd.to_datetime(comp.index, errors='coerce')
    (annual_returns.groupby(annual_returns['Year'].dt.year).prod() - 1).plot.bar()
    plt.title('Annual ret')
    #calplot.calplot(comp['Portfolio'])
    plt.figure()
    plt.title('Daily ret heatmap')
    df = comp['Portfolio'].to_frame().reset_index()
    df["Year"] = df.Dates.apply(lambda x: x.year)
    df["Month"] = df.Dates.apply(lambda x: x.strftime("%B"))
    df = df.pivot_table(index="Month",columns="Year",values="Portfolio", aggfunc='sum', fill_value=0)
    sns.heatmap(df, cmap="YlGnBu")
    
    # Batting average
    bat = comp['Portfolio']
    print('Win rate %: ', bat[bat>0].count()/bat.count())