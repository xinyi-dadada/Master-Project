import numpy as np
from igts import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import data_preprocess

def plot_segments(data, segments, name):
    plt.figure(figsize=(15, 6))

    for i in range(data.shape[0]):
        plt.plot(data[i], label=f'Channel {i+1}', alpha=0.7)

    for seg in segments:
        random_color = [random.random() for _ in range(3)]
        plt.axvline(x=seg, color=random_color, linestyle='--')

    plt.title('Time Series Segmentation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(name)


# prepare the np array for the data
def prepare_arr(data_s):
    """
    :param data_s: a dataframe
    :return: an array
    """
    all_col_l = []
    for col in col_names:
        df = data_s[col]
        data_series = pd.DataFrame(df)
        data_series_exploded = data_series.explode(col)
        merged_list = data_series_exploded[col].tolist()
        # normalize
        merged_l_mean = np.mean(merged_list)
        merged_l_std = np.std(merged_list)
        merged_list = (merged_list - merged_l_mean) / merged_l_std

        all_col_l.append(merged_list)

    all_col_arr = np.array(all_col_l)
    return all_col_arr

def igts_usage(arr, k, step, par_num, count):
    # IGTS
    TD_TT, IG_arr, knee = TopDown(arr, k, step=step)
    print(f'knee={knee}')
    k = knee
    TD_TT, IG_arr, knee = TopDown(arr, k, step=step)
    plot_segments(arr, TD_TT, f'/home/Shared/xinyi/blob1/thesis/figure/IGTS/IGTS_igcul_k={k}_s={step}_{par_num}_{count}')

def dynamic_programming_usage(arr, k, step):
    DP_TT, _ = DP_IGTS(arr, k, 100, 1)
    print('Dynamic Programming extracted TT >>>', DP_TT)

####
def rx_values(i_values, q_values):
    a1i = np.array([np.array(row) for row in i_values])
    a1q = np.array([np.array(row) for row in q_values])

    n = np.array(range(256)) / 256
    f_c = 24_000_000_000  # 24 GHz
    v = 2.0 * np.pi * f_c * n

    value_rx = a1i * np.cos(v) + a1q + np.sin(v)
    return value_rx, a1i, a1q

def prepare_arr(dt):
    rx1_a_i = dt['rx1_freq_a_channel_i_data']
    rx1_a_q = dt['rx1_freq_a_channel_q_data']
    rx2_a_i = dt['rx2_freq_a_channel_i_data']
    rx2_a_q = dt['rx2_freq_a_channel_q_data']
    rx1_b_i = dt['rx1_freq_b_channel_i_data']
    rx1_b_q = dt['rx1_freq_b_channel_q_data']

    # merge!
    a1_rx, a1_i_values, a1_q_values = rx_values(i_values=rx1_a_i,  q_values=rx1_a_q)
    a2_rx, a2_i_values, a2_q_values = rx_values(i_values=rx2_a_i,  q_values=rx2_a_q)
    b1_rx, b1_i_values, b1_q_values = rx_values(i_values=rx1_b_i,  q_values=rx1_b_q)

    #With the Merge function, now the channel of i and q are combined together, but it is a 2D array, need to reshape it to 1D
    a1_rx_1d = a1_rx.reshape(-1)
    a2_rx_1d = a2_rx.reshape(-1)
    b1_rx_1d = b1_rx.reshape(-1)

    #convert all three channels together and use information gain for segmentation
    merged_arrs = np.vstack((a1_rx_1d, a2_rx_1d, b1_rx_1d))

    return merged_arrs


#####
col_names = [
            'rx1_freq_a_channel_i_data', 'rx1_freq_a_channel_q_data',
            'rx2_freq_a_channel_i_data', 'rx2_freq_a_channel_q_data',
            'rx1_freq_b_channel_i_data', 'rx1_freq_b_channel_q_data'
            ]

# read parquet
dt = pd.read_parquet('/home/Shared/xinyi/blob1/thesis/data/parquet_samples/13_06_22/radar_samples_192.168.67.112_412.parquet')

dividedata = data.DivideData(dt)

participants, num_participants = dividedata.divide_participants()

for par in participants:
    participant = participants[par]
    par_col = participant.iloc[:, :6]
    par_arr = prepare_arr(par_col)

    #par_arr_split = np.split(par_arr, 10, axis=1)
    # start igts
    count = 1
    for arr in par_arr_split:
        igts_usage(arr, k=10, step=10, par_num=par, count=count)
        count += 1
    #dynamic_programming_usage(par_arr, k=10, step=1)


