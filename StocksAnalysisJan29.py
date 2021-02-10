#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 07:43:39 2021

@author: surya
o SE
1 LVGO
2 TENB
3 FTCH
4 SDC
5 PLUG
6 ADBE
7 VISA
8 NVTA
...
30 DIS

@author: surya rao
Variables from WGF stocks analysis group Agni Pariksha copyright Anil Bhojani
"""

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
#import plot_decision_regions as pdr
import pandas as pd
from io import StringIO

df = pd.read_csv("PortfolioAgniParikshaJan292021.csv",header=None)
df.columns =['Ticker','Outcome','Disruptor','Moat','CEO','USBased','TAM','MarketShare'
             ,'Usage','PriorRevGrowth','FutureRevGrowth','EarningsGrowth'
             ,'SizeAndTime','International','Expert','TipsRank','RuleBreaker'
             ,'GrowthPS','FounderStake','InstStake','Phase','ShortInterest']
print(df['Outcome'])
#Ydata =df['Outcome'].values
Ydata=df.iloc[1:31,1].values
Xdata =df.iloc[1:31,2:].values
lr = LogisticRegression(penalty ='l2', C=1.0)
lr.fit(Xdata,Ydata)
print(lr.coef_)
coeffecients = pd.DataFrame(lr.coef_,columns =df.columns[2:])
coeffecients.plot(kind="barh",figsize=(20,20), legend = 'reverse',sort_columns = True)

coeffecients.to_csv('/Users/surya/Documents/Training/MachineLearning/Code/Python/Stocks Analysis/coeff.csv', index=False)
