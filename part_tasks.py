import os
import glob
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.fft import fft
from data_preprocess import DivideData

class ProcessRawData():
    def __init__(self, radar_number):
        self.radar_number = radar_number
        self.tasks = pd.read_csv('/home/Shared/xinyi/blob1/thesis/data/task.csv')
        self.base_dir = '/home/Shared/xinyi/blob1/thesis/data/parquet_samples/'
        #self.file_pattern = os.path.join(base_dir, '**', f'radar_samples_192.168.67.{self.radar_number}*')

    def divide_task_index(self, participant):
        """
        for each participant df, annotate by tasks
        task info: T_1: uuid, T_2: uuid, ...
        :param participant: df for one participant
        :return: dictionary of {T_1: [position], T_2: [position], ...}
        """
        tasks = self.tasks
        task_index = {}
        # need to change the number of range according to the task number
        for i in range(11):
            T = participant[participant['task_uuid'] == tasks['UUID'][i]].index
            T = np.array(T) * 256
            task_index[f'T_{i}'] = T

        try:
            task_max = {key: value[-1] for key, value in task_index.items()}
            task_min = {key: value[0] for key, value in task_index.items()}
            merged_ind = {key: (task_min[key], task_max[key]) for key in task_index.keys()}
            return merged_ind
        except:
            print("cannot find task index")

    def apply_fft(self, row):
        fft_result = fft(row)
        fft_magnitude = np.abs(fft_result)
        fft_magnitude_no_dc = fft_magnitude[1:len(fft_result) // 2]
        return fft_magnitude_no_dc
    def prepare_data_cnn(self):
        file_pattern = os.path.join(self.base_dir, '**', f'radar_samples_192.168.67.{self.radar_number}*')
        # get all file names for this radar
        matching_files = glob.glob(file_pattern, recursive=True)
        part_task_all = []
        tasks = self.tasks
        for data_path in matching_files:
            divide_data = DivideData(data_path)
            part, num = divide_data.divide_participants()
            for key, value in enumerate(part):
                data_part = part[value]
                data_part = data_part.reset_index(drop=True)
                task_index = self.divide_task_index(data_part)
                cols = ['rx1_freq_a_channel_i_data', 'rx1_freq_a_channel_q_data',
                       'rx2_freq_a_channel_i_data', 'rx2_freq_a_channel_q_data',
                       'rx1_freq_b_channel_i_data', 'rx1_freq_b_channel_q_data', 'fft']
                num_tasks = 11
                task_arr = np.empty((num_tasks, len(cols)), dtype=object)
                try:
                    for key, value in task_index.items():
                        start, end = value
                        task_num = int(key.split('_')[1])
                        for i in range(len(cols)):
                            col = cols[i]
                            data_series = data_part[col]
                            data_series_exploded = data_series.explode(col)
                            merged_list = data_series_exploded.tolist()
                            task_l = merged_list[start:(end + 1)]
                            task_arr[task_num, i] = task_l
                except Exception as e:
                    print(f"{value}")

                if task_arr.any():
                    df_new = pd.DataFrame(task_arr)
                    df_new['task'] = df_new.index
                    rename_dict = {i: cols[i] for i in range(len(cols))}
                    df_new.rename(columns=rename_dict, inplace=True)
                    #df_new['fft_result'] = df_new['rx1_freq_a_channel_i_data'].apply(apply_fft)
                    part_task_all.append(df_new)

        df_part_task_all = pd.concat(part_task_all, ignore_index=True)
        df_part_task_all.to_parquet(f'/home/Shared/xinyi/blob1/thesis/radar_{self.radar_number}/all_part_tasks_1611.parquet')
        print('wengweng~~')
# input the radar number for dividing
process_data = ProcessRawData(114)
process_data.prepare_data_cnn()
#base_dir = '/home/Shared/xinyi/blob1/thesis/data/parquet_samples/'
#radar_no = 112
#file_pattern = os.path.join(base_dir, '**', f'radar_samples_192.168.67.{radar_no}*')