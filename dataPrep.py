from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
import datetime

file_raw = "../data/raw.csv"
file_firenj = "../data/FireNJ_address.csv"
file_climate = "../data/noaa_climate.csv"
testSetRatio = 0.1

df_raw = pd.read_csv(file_raw, usecols=[1, 2])
df_fireNJ = pd.read_csv(file_firenj, usecols=[0, 4, 5])
merge_fireNJ = pd.merge(left=df_fireNJ, right=df_raw, how='left', left_on='ID', right_on='ID')
merge_fireNJ['QHSJ'] = pd.to_datetime(merge_fireNJ.QHSJ).dt.strftime('%m/%d/%Y')

# Count fires for each day
df_dateCount = merge_fireNJ.QHSJ.value_counts().reset_index()
df_dateCount.columns = ['QHSJ', 'count']

# Read climate data
df_noaa = pd.read_csv(file_climate, usecols=[5, 6, 10])
df_noaa.fillna(0, inplace=True)
df_noaa['DATE'] = pd.to_datetime(df_noaa.DATE).dt.strftime('%m/%d/%Y')

# Merge climate date to fire count
df_all = pd.merge(left=df_noaa, right=df_dateCount, how='left', left_on='DATE', right_on='QHSJ')
df_all['DATE'] = pd.to_datetime(df_all['DATE'])
df_all = df_all[(df_all['DATE'] > datetime.date(2007, 1, 1)) & (df_all['DATE'] < datetime.date(2017, 10, 16))]
df_all.fillna(0, inplace=True)
df_all.drop(df_all.columns[[3]], axis = 1, inplace = True)
df_all['count(t-1)'] = df_all['count'].shift(1)
df_all.dropna(inplace=True)
df_all.set_index('DATE')

# Save train and test data file
pos_test = int(len(df_all) * (1 - testSetRatio))
df_train, df_test = df_all[:pos_test], df_all[pos_test:]
df_train.to_csv('./data_train.csv')
df_test.to_csv('./data_test.csv')

# Visualize the data
plt.figure(1)
plt.subplot(211)
plt.plot(df_train['DATE'], df_train['count'])
plt.subplot(212)
plt.plot(df_test['DATE'], df_test['count'])
plt.show()

print(df_all)

