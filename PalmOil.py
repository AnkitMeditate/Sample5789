# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 15:54:19 2021

@author: AN20157679
"""


import pandas as pd

import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from statsmodels.api import OLS

path = r'C:\Ankit\Personal\WOrk\assesment\ACE\PreScreen_r3\\'

filepath = path + 'palm_ffb.csv'

df = pd.read_csv(filepath)

df.head()

model = LinearRegression()

y = df.pop('FFB_Yield')
x = df
x['month'] = pd.to_datetime(df['Date']).dt.month
x = x.drop(['Date','time'],axis = 1)

model = OLS(y,x).fit()
model.summary()


