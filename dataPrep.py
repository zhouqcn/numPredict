from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
import datetime

def dateparse(timestamp):
    return pd.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

df_raw = pd.read_csv("../data/raw.csv", usecols=[1, 2])
df_fireNJ = pd.read_csv("../data/FireNJ_address.csv", usecols=[0, 4, 5])
merge_fireNJ = pd.merge(left=df_fireNJ, right=df_raw, how='left', left_on='ID', right_on='ID')
merge_fireNJ['QHSJ'] = pd.to_datetime(merge_fireNJ.QHSJ).dt.strftime('%m/%d/%Y')

# Count fires for each day
df_dateCount = merge_fireNJ.QHSJ.value_counts().reset_index()
df_dateCount.columns = ['QHSJ', 'count']

# Read climate data
df_noaa = pd.read_csv("../data/noaa_climate.csv", usecols=[5, 6, 10])
df_noaa.fillna(0, inplace=True)
df_noaa['DATE'] = pd.to_datetime(df_noaa.DATE).dt.strftime('%m/%d/%Y')

# merge climate date to fire count
df_all = pd.merge(left=df_noaa, right=df_dateCount, how='left', left_on='DATE', right_on='QHSJ')
df_all['DATE'] = pd.to_datetime(df_all['DATE'])
df_all = df_all[(df_all['DATE'] > datetime.date(2007, 1, 1)) & (df_all['DATE'] < datetime.date(2017, 10, 16))]




print(df_all)

