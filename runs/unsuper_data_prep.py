import os
import glob
import scipy
import numpy as np
import pandas as pd

roi_task = ['T_0', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'T_8', 'T_9', 'T_10']
col_names = ['rx1_freq_a_channel_i_data', 'rx1_freq_a_channel_q_data']
             #'rx2_freq_a_channel_i_data', 'rx2_freq_a_channel_q_data',
             #'rx1_freq_b_channel_i_data', 'rx1_freq_b_channel_q_data', 'fft']
part_all_tasks = []
part_1 = part['participant_3']
part_1 = part_1.reset_index(drop=True)
task_index = unsupervied_classify.divide_task_index(part_1)
col = 'rx1_freq_a_channel_i_data'
task_arr = np.empty((11, 1), dtype=object)
for key, value in task_index.items():
    if key in roi_task:# key: T_0, T_1, ...
        start, end = value
        task_num = int(key.split('_')[1])
        data_series = part_1[col]
        data_series_exploded = data_series.explode(col)
        merged_list = data_series_exploded.tolist()
        task_l = merged_list[start:(end+1)]
        task_arr[task_num, 0] = task_l