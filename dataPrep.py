from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time

def dateparse(timestamp):
    return pd.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

df_raw = pd.read_csv("./raw.csv", usecols=[1, 2])
df_fireNJ = pd.read_csv("./FireNJ_address.csv", usecols=[0, 4, 5])
merge_fireNJ = pd.merge(left=df_fireNJ, right=df_raw, how='left', left_on='ID', right_on='ID')            

print(merge_fireNJ)

