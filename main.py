# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 03:34:33 2021

@author: Wu
"""

import pandas as pd
import numpy as np
import dateutil.rrule as dr
from dateutil.relativedelta import relativedelta as drel
from pandas.tseries.offsets import BDay
import util
import generator
import math

def rebalancing(test_ind, test_weights, cap=1):
    w = 0
    watchlist=[]
    for t in test_ind.rank(ascending=False).dropna().sort_values().index:
        watchlist.append(t)
        if (w+test_weights[t])<cap:
            w+=test_weights[t]
        else:
            test_weights[t] = cap-w
            break
    
    test_weights[~test_weights.index.str.contains('|'.join(watchlist), na=False)] = 0
    test_weights[test_weights<0.02] = 0
    return test_weights

def missing_dates(calendar, close):
    filler = []
    for t in calendar:
        i=0
        while i == 0:
            if t in close_price.index:
                filler.append(t)
                i+=1
            else:
                t = t + BDay(-1)
    return pd.DatetimeIndex(filler)

def replacer(old, new):
    temp = new[new.isnull()].index
    old[temp] = 0
    return old

def weekly_monitoring(test_ind, test_weights, existing_weights):

    balance = 1 - existing_weights.sum()
    
    for t in test_ind.rank(ascending=False).dropna().sort_values().index:
        #watchlist.append(t)
        if (existing_weights[t] >= test_weights[t]):
            continue
        elif (balance - (test_weights[t] - existing_weights[t]))>0:
            existing_weights[t] = test_weights[t]
            balance -= (test_weights[t] - existing_weights[t])
        else:
            existing_weights[t] += balance
            break
    
    return existing_weights

#np.random.seed(10)
close_price = pd.read_csv("close.csv", index_col="Dates", parse_dates=True)
open_price = pd.read_csv("open prices.csv", index_col="Dates", parse_dates=True)

# Signal generated using generator.py
mom_index_90 = pd.read_csv("mom_open_90.csv", index_col="Dates", parse_dates=True)

PIT = pd.read_csv("PIT.csv", index_col="Date", parse_dates=True)
klci = pd.read_csv("klci div adj.csv", index_col="Date", parse_dates=True)
# Data on bid-ask spread only go as far back as 3 Jan 2012
spread = pd.read_csv("bidask.csv", index_col="Dates", parse_dates=True).reindex(open_price.index).ffill().bfill()


# Create rebalancing schedule
first_month = close_price.index[0]
last_month = close_price.index[-1] + drel(months=0, days=+1)
fortnight = pd.DatetimeIndex(dr.rrule(dr.WEEKLY, interval = 3, byweekday=(dr.WE),dtstart=first_month, until=last_month))
fortnight = missing_dates(fortnight, close_price)

weekly = pd.DatetimeIndex(dr.rrule(dr.WEEKLY, byweekday=(dr.WE), dtstart=first_month, until=last_month))
weekly = missing_dates(weekly, close_price)

# Only buy if index is above its 200 day average
index = klci['PX_OPEN']/klci['PX_OPEN'].rolling(200).mean()
filter_index = index.mask(index>1,1).mask(index<=1,np.nan)

# Only buy stocks with no recent 1-day % gap exceeding 15%
temp = open_price.pct_change().abs()
temp = temp.mask(temp>=0.15,np.nan).mask(temp<=0.15,0)
filter_recentgap = temp.rolling(10).sum(skipna=False).replace(0,1)   # Sharpe higher without this

# Only buy stocks above its 100 day moving average
temp = open_price/open_price.rolling(100).mean()
filter_100d = temp.mask(temp>1,1).mask(temp<=1,np.nan)

weights = 0.001 / open_price.pct_change().ewm(adjust=True, span=36, min_periods=36).std() # 0.001 works out to be 25% portfolio volatility
weights = weights.clip(0,0.1)

adj_database = mom_index_90 * PIT * filter_100d * filter_recentgap #* filter_index_rank

ftx = pd.DataFrame(np.nan, index=close_price.index, columns=close_price.columns)

def inertia(a,b):
    inert = a-b
    inert = inert.mask((inert>=0.02)|(inert<=-0.02), 0)

    return inert

for i,j in enumerate(fortnight):  #weekly
    if filter_index.loc[j] == 1:
        ftx.loc[j] = rebalancing(adj_database.loc[j], weights.loc[j])
    else:
        ftx.loc[j] = replacer(ftx.loc[fortnight[i-1]], adj_database.loc[j])  #ftx.loc[weekly[i-1]]

    ftx.loc[j] -= inertia(ftx.loc[j], ftx.loc[fortnight[i-1]])

ftx.ffill(inplace=True)

def cl_tweak(cl, op, schedule, ftx):

    cl_chg = cl.pct_change()
    
    z = ftx.diff(-1)
    overnight_filter = z.mask(z>0, 1).mask(z<=0,0).fillna(0)

    overnight_ret = (op.shift(-1) - cl) / cl
    intraday = ((cl-op)/op).fillna(0)

    overnight_ret = overnight_ret * overnight_filter

    for k in schedule:
        cl_chg.loc[k] = intraday.loc[k]

    return cl_chg + overnight_ret.fillna(0)

adj_cl = cl_tweak(close_price, open_price, fortnight, ftx)

test = ftx.diff().abs().fillna(0)
ftx = (adj_cl*ftx).sum(axis=1).rename('Gross').to_frame()
print('Pre cost Sharpe ratio: ', np.sqrt(250)*np.nanmean(ftx.values - klci['PX_LAST'].pct_change().values)/np.nanstd(ftx.values - klci['PX_LAST'].pct_change().values))


def roundup(x):
    return int(math.ceil(float(x) / 1000.0))

def brokerage_calculator(amt):
    # https://www.bursamalaysia.com/trade/trading_resources/equities/transaction_costs
    if (amt == 0) or (amt == np.nan):
        return 0
    else:
        float_brokerage = 0.0008*amt
        fixed_brokerage = 8    # Rakuten Trade round trip RM7; HL 2*RM8
        stampduty = min(200, roundup(amt))
        clearingfee = min(1000, 0.0003*amt)
        
        return (max(fixed_brokerage,float_brokerage) + stampduty + clearingfee)

def brokerage(u, v, w, z):
    temp = u*v

    fees = 0
    for x in temp:
        fees+=brokerage_calculator(x)
        
    return fees + (np.nansum(u*v*z/w)/2)
    
def transform(dr, stocks, P, bidaskspread):
    capital = 80000
    start = np.zeros_like(dr)
    fee = np.zeros_like(dr)
    end = np.zeros_like(dr)

    start[0] = capital
    fee[0] = 0
    end[0] = (start[0] * (1+dr[0])) - fee[0]
    
    for idx in range(1, len(dr)):
        start[idx] = end[idx-1]
        fee[idx] = brokerage(start[idx], stocks[idx], P[idx], bidaskspread[idx])
        end[idx] = (start[idx] * (1+dr[idx])) - fee[idx]
        
    return start, end, fee  

ftx['Start'], ftx['End'], ftx['Fee'] = transform(ftx.values, [test.iloc[i].values for i in range(len(test))],\
                                                 [open_price.iloc[i].values for i in range(len(open_price))], [spread.iloc[i].values for i in range(len(spread))])
#ftx.to_csv('dailytradedata.csv')

comp = pd.DataFrame(data={'Portfolio': ftx['End'].pct_change(), 'Benchmark': klci['PX_LAST'].pct_change()}, index=ftx.index)

util.visualization(comp)