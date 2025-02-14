### prepare the dataset for
import glob
import pandas as pd
import numpy as np
import os

col_names = ['rx1_freq_a_channel_i_data', 'rx1_freq_a_channel_q_data',
             'rx2_freq_a_channel_i_data', 'rx2_freq_a_channel_q_data',
             'rx1_freq_b_channel_i_data', 'rx1_freq_b_channel_q_data',
             'rx_1_a','rx_1_b','rx_2_a','fft','Task', 'Participant', 'Radar']
radar_data = pd.DataFrame(columns=col_names)

radar_name = 'radar_112'
tasks = [0, 6, 7]
#radar_name = 'radar_114'
#tasks = []

base_dir = '/home/Shared/xinyi/blob1/thesis/data/parquet_samples/'
file_pattern = os.path.join(base_dir, '**', 'radar_samples_192.168.67.112*')

matching_files = glob.glob(file_pattern, recursive=True)

part_no = 0

for file_path in matching_files:

    # get the number of participants
    par_num = len(os.listdir(file_path))
    for j in range(1, par_num+1):
        for i in tasks:
            # Add label for each dataset, task and participant number
            data = pd.read_parquet(f"{file_path}/part_{j}/task_{i}.parquet")
            data['Task'] = i
            data['Participant'] = j + part_no
            data['Radar'] = radar_name
            radar_data = pd.concat([radar_data, data], ignore_index=True)
            #data.to_parquet(f'{file_path}/part_{j}/task_{i}.parquet')
    part_no = part_no + par_num
    print(part_no)

radar_data.to_parquet('radar_112/radar_112_all.parquet')
"""
for j in range(1, 4): # only three participants now
    for i in range(0, 7):
        data_new = pd.read_parquet(f"{file_path}/part_{j}/task_{i}.parquet")
        # merge all the data in radar 44 to one parquet
        if i == 0 and j == 1:
            continue
        radar_data = pd.concat([radar_data, data_new], ignore_index=True)
radar_data.to_parquet(f'{file_path}/radar_data.parquet')




data_1 = pd.read_parquet('data/parquet_samples/radar_samples_192.168.67.114_410.parquet')
print(data_1.shape)
data_3 = pd.read_parquet('13_06_22/radar_114/radar_114_data.parquet')
print(data_3.shape)
"""
