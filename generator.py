# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:39:33 2021

@author: Wu
"""

import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS

def signal_generator(raw_data, window_length=90):
    
    data=np.log(raw_data)
    data.ffill(inplace=True)
    names = data.columns
    
    rsquared = data.copy()
    slope = data.copy()
    
    data.loc[:, 'const'] = 1
    data['X'] = np.arange(1,len(data)+1)
    
    for i in names:
        model = RollingOLS(endog = data.loc[:, i].values , exog=data.loc[:, ['const','X']], window=window_length)
        rres = model.fit()
        annualized_slope = ((np.exp(rres.params.loc[:, 'X']))**250) - 1
        
        slope.loc[:, i] = annualized_slope
        rsquared.loc[:, i] = rres.rsquared 

    temp = slope*rsquared
    return temp

if __name__ == '__main__':
    raw_data = pd.read_csv("open prices.csv", index_col="Dates", parse_dates=True)
    
    lookback = 90
    
    temp = signal_generator(raw_data, lookback)
    
    temp.to_csv('mom_open_{0}.csv'.format(str(lookback)))