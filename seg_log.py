import numpy as np
import pandas as pd
from data_preprocess import DivideData
import re
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import random


class SegLogFunction():
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.important_task = ['T_0', 'T_6', 'T_7', 'T_11', 'T_12']

    def log_func(self, signal, A, B):
        f_radarsignal = 1 / (1 + np.exp((-A * (signal - B))))
        return f_radarsignal

    def rx_values(self, i_values, q_values):
        """
        use to merge the i and q channels
        :param i_values: e.g.'rx1_freq_a_channel_i_data'
        :param q_values: e.g.'rx1_freq_a_channel_q_data'
        :return: merged channels and i and q channels (flattened)
        """
        a1i = np.array([np.array(row) for row in i_values])
        a1q = np.array([np.array(row) for row in q_values])

        n = np.array(range(256)) / 256
        f_c = 24_000_000_000  # 24 GHz
        v = 2.0 * np.pi * f_c * n

        value_rx = a1i * np.cos(v) + a1q + np.sin(v)
        return value_rx, a1i, a1q

    def prepare_merged_channels(self, dt):
        rx1_a_i = dt['rx1_freq_a_channel_i_data']
        rx1_a_q = dt['rx1_freq_a_channel_q_data']
        rx2_a_i = dt['rx2_freq_a_channel_i_data']
        rx2_a_q = dt['rx2_freq_a_channel_q_data']
        rx1_b_i = dt['rx1_freq_b_channel_i_data']
        rx1_b_q = dt['rx1_freq_b_channel_q_data']
        # merge!
        a1_rx, a1_i_values, a1_q_values = self.rx_values(i_values=rx1_a_i, q_values=rx1_a_q)
        a2_rx, a2_i_values, a2_q_values = self.rx_values(i_values=rx2_a_i, q_values=rx2_a_q)
        b1_rx, b1_i_values, b1_q_values = self.rx_values(i_values=rx1_b_i, q_values=rx1_b_q)
        # return a1_rx, a2_rx, b1_rx
        a1_rx_nested_arr = a1_rx.tolist()
        a2_rx_nested_arr = a2_rx.tolist()
        b1_rx_nested_arr = b1_rx.tolist()

        return a1_rx_nested_arr, a2_rx_nested_arr, b1_rx_nested_arr

    def flatten_df(self, df, start, end):
        df = df.iloc[:, :7]

        a1_rx, a2_rx, b1_rx = self.prepare_merged_channels(df)
        df['a1_rx'] = a1_rx
        df['a2_rx'] = a2_rx
        df['b1_rx'] = b1_rx
        task_arr = np.empty((10, 1), dtype=object)

        k = 0
        for col in df.columns:
            data_series = df[col]
            data_series_exploded = data_series.explode(col)
            merged_list = data_series_exploded.tolist()
            task_l = merged_list[start:(end + 1)]
            task_arr[k, 0] = task_l
            k += 1
        df_new = pd.DataFrame(task_arr.T, columns=df.columns)
        return df_new

    def get_task_index(self, data_part1):
        task_and_split = {}
        task_index = divide_data.divide_task_index(data_part1)
        # here is for radar 112, the feature tasks are task 0, 6, 7, 11, 12
        important_task = self.important_task
        for task_name in important_task:
            try:
                start, end = task_index[task_name]
                label = re.search(r'\d+', task_name)
                label = int(label.group())
                task_and_split[task_name] = [(start, end), label]
            except:
                continue
        combined_dict = {}
        data = task_and_split

        # combine T_6 and T_7
        if 'T_6' in data and 'T_7' in data:
            combined_dict['split_2'] = [
                (data['T_6'][0][0], data['T_7'][0][1]),  # Combine ranges
                [data['T_6'][1], data['T_7'][1]]  # Combine labels
            ]
        # same for combining T_11 and T_12
        if 'T_11' in data and 'T_12' in data:
            combined_dict['split_3'] = [
                (data['T_11'][0][0], data['T_12'][0][1]),
                [data['T_11'][1], data['T_12'][1]]
            ]
        # for T_0
        try:
            combined_dict['split_1'] = [data['T_0'][0], [data['T_0'][1]]]
        except:
            print('There is no Task 0')

        return combined_dict

    def split_and_labels(self, task_and_split, input_index):
        for key, (truth_index, label) in task_and_split.items():
            if (input_index[0] <= truth_index[1] and input_index[1] >= truth_index[0]):
                return label

    def seg_part(self, data_part1, task_and_split, name=None):
        radarsignal = divide_data.process_each_participant(data_part1)
        radarsignal = radarsignal.astype(np.float64)
        A = 0.01
        q = 0.001
        B = np.quantile(radarsignal, q=q)
        f_radarsignal = self.log_func(radarsignal, A, B)
        f_radarsignal = 1 - np.squeeze(f_radarsignal)

        radarsignal = np.squeeze(radarsignal)
        x_axis_seconds = np.arange(len(radarsignal)) * 0.23 * (1 / 256)

        feature_index = np.where(f_radarsignal > 0.005)[0]
        buffer = 8000
        feature_split = {}
        index_num = feature_index[0]
        split_num = 1  # for name the split in feature_split
        for i in range(1, len(feature_index)):
            if feature_index[i] - feature_index[i - 1] > 10000:
                # max() in case the index start from 0, and the end will not reach the end of the signal, so it's safe
                feature_split[f'split_{split_num}'] = [max(index_num - buffer, 0), feature_index[i - 1] + buffer]
                split_num += 1
                index_num = feature_index[i]

        all_split = []
        for key, value in enumerate(feature_split):
            start, end = feature_split[value]
            data_part = self.flatten_df(data_part1, start, end)
            label = self.split_and_labels(task_and_split, (start, end))
            if label is not None:
                data_part['label'] = None
                data_part.at[0, 'label'] = label
                all_split.append(data_part)
        # plot: signal split by task id, result of logistic regression function, signal split by thresholding
        task_index = divide_data.divide_task_index(data_part1)

        plt.figure(figsize=(12, 16))
        sns.set(style='whitegrid')
        sns.color_palette("pastel")
        plt.subplots_adjust(hspace=0.5)
        # plot the original signal and the true tasks
        plt.subplot(3, 1, 1)
        sns.lineplot(x=x_axis_seconds, y=radarsignal, color='grey', alpha=0.8)
        for task_name in self.important_task:
            start, end = task_index[task_name]
            start *= 0.23 * (1 / 256)
            end *= 0.23 * (1 / 256)
            random_color = (random.random(), random.random(), random.random())
            plt.axvspan(start, end, color=random_color, alpha=0.3, label=task_name)

        plt.legend(loc='lower right')
        plt.title('a', loc='left', fontsize=20)
        plt.xlabel('sec', loc='right', fontsize=15)
        # plot the resulf of logistic regression
        plt.subplot(3, 1, 2)
        sns.lineplot(x=x_axis_seconds, y=f_radarsignal, color='lightgreen', alpha=0.5)
        plt.title('b', loc='left', fontsize=20)
        plt.xlabel('sec', loc='right', fontsize=15)
        # plot the segmented feature parts
        plt.subplot(3, 1, 3)
        sns.lineplot(x=x_axis_seconds, y=radarsignal, color='grey', alpha=0.8)
        for key, value in enumerate(feature_split):
            start, end = feature_split[value]
            start *= 0.23 * (1 / 256)
            end *= 0.23 * (1 / 256)
            random_color = (random.random(), random.random(), random.random())
            plt.axvspan(start, end, color=random_color, alpha=0.3, label=value)
        plt.title('c', loc='left', fontsize=20)
        plt.xlabel('sec', loc='right', fontsize=15)
        plt.legend(loc='lower right')
        plt.show()
        plt.clf()
        return all_split
"""
    def seg_part(self, data_part1, task_and_split, name=None):
        radarsignal = divide_data.process_each_participant(data_part1)
        radarsignal = radarsignal.astype(np.float64)
        A = 0.01
        q = 0.001
        B = np.quantile(radarsignal, q=q)
        f_radarsignal = self.log_func(radarsignal, A, B)
        f_radarsignal = 1 - np.squeeze(f_radarsignal)

        radarsignal = np.squeeze(radarsignal)
        x_axis_seconds = np.arange(len(radarsignal)) * 0.23

        feature_index = np.where(f_radarsignal > 0.005)[0]
        buffer = 8000
        feature_split = {}
        index_num = feature_index[0]
        split_num = 1  # for name the split in feature_split
        for i in range(1, len(feature_index)):
            if feature_index[i] - feature_index[i - 1] > 10000:
                # max() in case the index start from 0, and the end will not reach the end of the signal, so it's safe
                feature_split[f'split_{split_num}'] = [max(index_num - buffer, 0), feature_index[i - 1] + buffer]
                split_num += 1
                index_num = feature_index[i]

        all_split = []
        for key, value in enumerate(feature_split):
            start, end = feature_split[value]
            data_part = self.flatten_df(data_part1, start, end)
            label = self.split_and_labels(task_and_split, (start, end))
            if label is not None:
                data_part['label'] = None
                data_part.at[0, 'label'] = label
                all_split.append(data_part)
        
        # plot: signal split by task id, result of logistic regression function, signal split by thresholding
        task_index = divide_data.divide_task_index(data_part1)
        plt.figure(figsize=(12, 16))
        sns.set(style='whitegrid')
        sns.color_palette("pastel")
        plt.subplots_adjust(hspace=0.5)
        # plot the original signal and the true tasks
        plt.subplot(3, 1, 1)
        sns.lineplot(x=x_axis_seconds, y=radarsignal, color='grey', alpha=0.8)
        for task_name in self.important_task:
            try:
                start, end = task_index[task_name]
                start *= 0.23
                end *= 0.23
                random_color = (random.random(), random.random(), random.random())
                plt.axvspan(start, end, color=random_color, alpha=0.3, label=task_name)
            except:
                continue

        plt.legend(loc='lower right')
        plt.title('a', loc='left', fontsize=20)
        plt.xlabel('sec', loc='right', fontsize=15)
        # plot the resulf of logistic regression
        plt.subplot(3, 1, 2)
        sns.lineplot(x=x_axis_seconds, y=f_radarsignal, color='lightgreen', alpha=0.5)
        plt.title('b', loc='left', fontsize=20)
        plt.xlabel('sec', loc='right', fontsize=15)
        # plot the segmented feature parts
        plt.subplot(3, 1, 3)
        sns.lineplot(x=x_axis_seconds, y=radarsignal, color='grey', alpha=0.8)
        for key, value in enumerate(feature_split):
            start, end = feature_split[value]
            start *= 0.23
            end *= 0.23
            random_color = (random.random(), random.random(), random.random())
            plt.axvspan(start, end, color=random_color, alpha=0.3, label=value)
        plt.title('c', loc='left', fontsize=20)
        plt.xlabel('sec', loc='right', fontsize=15)
        plt.legend(loc='lower right')
        plt.savefig(f'/home/Shared/xinyi/blob1/thesis/logs_seg/result{name}')
        plt.clf()
"""



# Set the base path, subfolders, and task filename
base_dir = '/home/Shared/xinyi/blob1/thesis/data/parquet_samples/'
seg_function = SegLogFunction(base_dir)
radar_no = 112
file_pattern = os.path.join(base_dir, '**', f'radar_samples_192.168.67.{radar_no}*')
# get all file names for this radar
matching_files = glob.glob(file_pattern, recursive=True)
all_split = []
tasks = pd.read_csv('/home/Shared/xinyi/blob1/thesis/data/task.csv')
### split by log_func ###
for data_path in matching_files:
    divide_data = DivideData(data_path)
    part, num = divide_data.divide_participants()
    for key, value in enumerate(part):
        data_part = part[value]
        data_part = data_part.reset_index(drop=True)
        task_and_split = seg_function.get_task_index(data_part)
        all_split_part = seg_function.seg_part(data_part, task_and_split)
        all_split += all_split_part

df_all_split = pd.concat(all_split, ignore_index=True)
save_path = '/home/Shared/xinyi/blob1/thesis/logs_seg/radar112_seg_all_2110.parquet'
df_all_split.to_parquet(save_path)


"""
# plot for the result
data_path = '/home/Shared/xinyi/blob1/thesis/data/parquet_samples/15_06_22/radar_samples_192.168.67.112_758.parquet'
divide_data = DivideData(data_path)
part, num = divide_data.divide_participants()
name = 1
for key, value in enumerate(part):
    data_part = part[value]
    data_part = data_part.reset_index(drop=True)
    task_and_split = seg_function.get_task_index(data_part)
    all_split_part = seg_function.seg_part(data_part, task_and_split, name)
    name += 1

"""