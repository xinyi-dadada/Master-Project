import numpy as np
from igts import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from data import DivideData

def igts_usage(arr, k, step):
    # IGTS
    TD_TT, IG_arr, knee = TopDown(arr, k, step=step)
    print(f'knee={knee}')
    k = knee
    TD_TT, IG_arr, knee = TopDown(arr, k, step=step)
    return TD_TT
    #plot_segments(arr, TD_TT, f'/home/Shared/xinyi/blob1/thesis/figure/IGTS/IGTS_igcul_k={k}_s={step}_{par_num}_{count}')

def dynamic_programming_usage(arr, k, step):
    DP_TT, _ = DP_IGTS(arr, k, 100, 1)
    print('Dynamic Programming extracted TT >>>', DP_TT)

dt = '/home/Shared/xinyi/blob1/thesis/data/parquet_samples/13_06_22/radar_samples_192.168.67.112_412.parquet'
divide_data = DivideData(dt)
participants, num_par = divide_data.divide_participants()
data_part = participants['participant_1']

i_values = data_part['rx1_freq_a_channel_i_data']
r1x_ai = a1i = np.array([np.array(row) for row in i_values])
r1x_ai = r1x_ai.flatten().reshape(-1, 1)

seg_result = igts_usage(r1x_ai, 7, 1000)

print(seg_result)
print('wengweng is checking')
