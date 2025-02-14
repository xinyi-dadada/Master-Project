# data preparation for segmentation cnn training
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from igts import *
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F  # Import for padding
#from data import DivideData

# Define a custom Dataset class to handle this structure
class ParticipantDataset(torch.utils.data.Dataset):
    def __init__(self, participant_datasets):
        self.participant_datasets = participant_datasets

    def __len__(self):
        return len(self.participant_datasets)

    def __getitem__(self, idx):
        return self.participant_datasets[idx]


class SegmentCNNPrepare():
    def __init__(self, data_path, important_task=None):
        self.dt = pd.read_parquet(data_path)
        self.important_tasks = important_task
        # self.radar = radar
        self.tasks = pd.read_csv('/home/Shared/xinyi/blob1/thesis/data/task.csv')
        # signal columns names
        self.signal_col_names = ['rx1_freq_a_channel_i_data', 'rx1_freq_a_channel_q_data',
                                 'rx2_freq_a_channel_i_data', 'rx2_freq_a_channel_q_data',
                                 'rx1_freq_b_channel_i_data', 'rx1_freq_b_channel_q_data',
                                 'a1_rx', 'a2_rx', 'b1_rx']
        self.torch_length = 1200
    # function for merge channel i and q
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

    # nomalize data
    def normalize_neg_pos(self, array):
        mean = np.mean(array)
        range_val = np.max(array) - np.min(array)
        return 2 * (array - mean) / range_val

    # prepare merged channels input
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
        #return a1_rx, a2_rx, b1_rx
        a1_rx_nested_arr = a1_rx.tolist()
        a2_rx_nested_arr = a2_rx.tolist()
        b1_rx_nested_arr = b1_rx.tolist()

        return a1_rx_nested_arr, a2_rx_nested_arr, b1_rx_nested_arr

    # use IGTS to find the position for segmentation
    def igts_usage(self, arr, k, step, par_num, count, maxig):
        # IGTS
        TD_TT, IG_arr, knee = TopDown(arr, k, step=step, maxIG=maxig)
        print(f'knee={knee}')
        k = knee
        TD_TT, IG_arr, knee = TopDown(arr, k, step=step, maxIG=maxig)

        return TD_TT, knee

    def plot_segments(self, data, segments, save_path):
        # used to visualize the segment result
        plt.clf()
        for i in range(data.shape[0]):
            plt.plot(data[i], label=f'Channel {i + 1}', alpha=0.7)

        for seg in segments:
            random_color = [random.random() for _ in range(3)]
            plt.axvline(x=seg, color=random_color, linestyle='--')

        plt.title('Time Series Segmentation')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def prepare_dataloader(self):
        # merged the corresponding i and q channels
        a1_rx, a2_rx, b1_rx = self.prepare_merged_channels(self.dt)
        self.dt['a1_rx'] = a1_rx
        self.dt['a2_rx'] = a2_rx
        self.dt['b1_rx'] = b1_rx
        # first initialize the label as 0
        self.dt['label'] = 0
        participants_datasets = []
        # divide data by participants
        divide_data = DivideData(self.dt, '/home/Shared/xinyi/blob1/thesis')
        participants, num_par = divide_data.divide_participants()
        print(f'number of participants: {num_par}')
        #batch_size = num_par

        sd_length = self.torch_length

        # prepare the data loader for each participant
        for par in participants:
            participant = participants[par]

            # prepare labels for supervised learning segmentation
            task_index = {}
            for i in range(15):
                T = participant[participant['task_uuid'] == self.tasks['UUID'][i]].index
                task_index[f'T_{i}'] = T
            # if the task is in important task, change the label to 1
            for task in self.important_tasks:
                participant.loc[task_index[task], 'label'] = 1

            # data and labels
            training_data_df, training_labels = participant.loc[:, self.signal_col_names], participant['label']
            #df = training_data_df.applymap(np.array)
            #df = df.T
            training_data_array = training_data_df.applymap(np.array).T.values.tolist()
            #training_data_array = training_data_array.T
            # the training data need to convert to a numpy array
            #training_data_array = np.array(df.values.tolist())
            # new axis for 256
            training_data = np.stack(training_data_array)
            train_data_tensor = torch.tensor(training_data, dtype=torch.float64)
            split_torch_data = torch.split(train_data_tensor, sd_length, dim=1)
            split_torch_data = list(split_torch_data)
            # Pad only the length (second dimension) to make all the participant has the same shape of data and label tensor
            # data
            current_length = split_torch_data[-1].shape[1]
            ###
            if current_length < sd_length:
                # Padding is applied only to the second dimension (length)
                padding = [0, 0,  # No padding for width (256)
                           0, sd_length - current_length]  # Padding for length (10000)

                # Apply padding
                train_data_tensor_padded = F.pad(train_data_tensor, padding)
            else:
                train_data_tensor_padded = train_data_tensor[:, :sd_length, :]
            ###
            # label
            train_labels_tensor = torch.tensor(training_labels.values, dtype=torch.int)
            split_torch_labels = torch.split(train_labels_tensor, sd_length)
            split_torch_labels = list(split_torch_labels)
            # Pad the labels to have a length of 10000
            #current_label_length = split_torch_labels[-1].shape[0]
            if current_length < sd_length:
                data_padding = padding = [0, 0,  # No padding for width (256)
                           0, sd_length - current_length]  # Padding for length to 1200
                split_torch_data[-1] = F.pad(split_torch_data[-1], data_padding)
                # Padding for labels (pad the second dimension, i.e., length)
                label_padding = [0, sd_length - current_length]  # Padding at the end
                split_torch_labels[-1] = F.pad(split_torch_labels[-1], label_padding, value=0)  # Padding with 0
            for i in range(len(split_torch_data)):
                participants_datasets.append((split_torch_data[i], split_torch_labels[i]))
        batch_size = len(participants_datasets)
        dataset = ParticipantDataset(participants_datasets)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader

