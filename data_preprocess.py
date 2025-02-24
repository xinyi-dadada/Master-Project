import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from sklearn.preprocessing import StandardScaler
from seg_cnn_prep import SegmentCNNPrepare


class DivideData():
    def __init__(self, data_path):
        # input data and tasks info
        self.data = pd.read_parquet(data_path)
        self.tasks = pd.read_csv('/home/Shared/xinyi/blob1/thesis/data/task.csv')
        self.folder = '/home/Shared/xinyi/blob1/thesis'
        self.data_path = data_path
        # random_colors = self.generate_random_color(self)

    # divide participants data based on timestamp
    def divide_participants(self):
        """
        There are several participants in one dataframe, need to divide them.
        Once the difference of timestamp is larger than 1 hour, showing that the people changed
        :return: dictionary: {'participant_1': [...], 'participant_2': [...] ...}
        """
        # compute the time difference
        time_difference = self.data['timestamp'].diff()
        self.data['timedifference'] = time_difference
        # if the time gap is more than one hour, participant changed
        divide_part = self.data[self.data['timedifference'] > pd.Timedelta(hours=1)]
        divide_points = list(divide_part.index)
        # prepare for dividing participants
        num_participants = len(divide_points) + 1
        participants = {}
        min_val = 0
        # participant_1: data[0:divide_points[0]]
        # participant_1: data[divide_points[0]:divide_points[1]]
        # ...
        # participant_lastone: data[divide_points[-1]:]
        for i in range(num_participants):
            if i != (num_participants - 1):
                max_val = divide_points[i]
                participants[f'participant_{i + 1}'] = self.data[min_val:max_val]
                min_val = max_val
            else:
                participants[f'participant_{i + 1}'] = self.data[min_val:]
        return participants, num_participants

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
        for i in range(15):
            T = participant[participant['task_uuid'] == tasks['UUID'][i]].index
            T = np.array(T) * 256
            task_index[f'T_{i}'] = T

        try:
            task_max = {key: value[-1] for key, value in task_index.items()}
            task_min = {key: value[0] for key, value in task_index.items()}
            merged_ind = {key: (task_min[key], task_max[key]) for key in task_index.keys()}
            return merged_ind
        except:
            print(participant)

    def igts_usage(self, arr, k, step, par_num=None, count=None):
        # IGTS
        TD_TT, IG_arr, knee = TopDown(arr, k, step=step)
        print(f'knee={knee}')
        k = knee
        TD_TT, IG_arr, knee = TopDown(arr, k, step=step)
       # plot_segments(arr, TD_TT,
                     # f'/home/Shared/xinyi/blob1/thesis/figure/IGTS/IGTS_igcul_k={k}_s={step}_{par_num}_{count}')

    def divide_task_parquet(self, selected_task=list):
        """
        for each participant, divide their task and save as a parquet file for further use
        :param participants: participants dictionary got from divide_participants, there are always four participants in one dictionary
        :return: store each task as a np arr with shape of (6, task_signal_len) -> npy per participant and plot
        """
        participants, num = self.divide_participants()
        print(f'participant number: {num}')
        col_names = self.data.columns[:6]
        for i in range(num):
            participant = participants[f'participant_{i + 1}']
            participant = participant.reset_index(drop=True)

            # divide data by task per participant, e.g. participant 1, return the first and the last index of this task
            task_all = self.divide_task_index(participant)
            selected_task = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            #for T0 to T6, store the signal for each task
            for j in selected_task:
                start, end = task_all[f'T_{j}']
                # prepare an array for storing signal data for 6 channels
                task_arr = np.empty((6, end-start))
                k = 0 # initialize for channels
                for col in col_names:
                    # exploded the 2D array to a 1D list, for plotting and computation
                    data_series = participant[col]
                    data_series_exploded = data_series.explode(col)
                    merged_list = data_series_exploded.tolist()
                    task_l = merged_list[start:end]
                    # normalize
                    task_l_mean = np.mean(task_l)
                    task_l_std = np.std(task_l)
                    task_arr[k,:] = (task_l - task_l_mean) / task_l_std
                    # tried StandardScaler but not work well
                    # task_arr[k,:] = StandardScaler().fit_transform(merged_list)
                    # next channel
                    k += 1

                # set file path
                check_path = f'{self.folder}/part_{i + 1}'
                if not os.path.exists(check_path):
                    os.makedirs(check_path)
                file_path = f'{self.folder}/part_{i + 1}/task_{j}'
                df = pd.DataFrame(task_arr.T, columns=col_names)
                df.to_parquet(f'{file_path}.parquet', engine='fastparquet')

    def signal_process(self, arr, part_num=None):

        df = pd.DataFrame({
            'a1_rx': arr
        })
        df_exploded = df.explode('a1_rx')
        #exploded_list = df_exploded.tolist()

        s = np.array(df_exploded, dtype=np.float64)
        #s = np.sqrt(s)
        s_hat = s - np.mean(s)
        #s_hat_hat = s_hat + np.abs(np.min(s_hat))
        return s_hat

    def process_each_participant(self, one_participant):
        segprepare = SegmentCNNPrepare(data_path=self.data_path)
        # get the task info for this part

        one_participant = one_participant.reset_index(drop=True)
        #task_part = self.divide_task_index(one_participant)
        one_participant = one_participant.iloc[:, :6]
        a1_rx, a2_rx, b1_rx = segprepare.prepare_merged_channels(one_participant)
        merged_df = pd.DataFrame({
            'a1_rx': a1_rx,
            'a2_rx': a2_rx,
            'b1_rx': b1_rx
        })
        s_hat_hat = self.signal_process(a1_rx)
        return s_hat_hat

    def plot_figure(self, df, title, task, num=None):
        data_series = pd.DataFrame(df)
        data_series_exploded = data_series.explode(title)
        merged_list = data_series_exploded[title].tolist()

        # only for one task
        for key, (min_val, max_val) in task.items():
            random_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            plt.plot(merged_list[min_val:max_val])
            plt.hlines(y=30000, xmin=min_val, xmax=max_val, colors=random_color, linewidth=1, label=key)
        plt.legend()
        plt.title(title)
        plt.show()
        plt.savefig(f'figure/participant{num + 1}_{title}')
        plt.close()

    def plot_figure_task(self, df, title, task, num):
        """
        plot and store as npy for each task per participant
        :param df: only have one of col_name
        :param title: col_name
        :param task: task.csv
        :param num: number of participant
        :return: figure and save each task: a npy file contains all 6 channels for this task
        """
        data_series = pd.DataFrame(df)
        data_series_exploded = data_series.explode(title)
        merged_list = data_series_exploded[title].tolist()

        # only for one task
        for key, (min_val, max_val) in task.items():
            plt.plot(merged_list[min_val:max_val])
            plt.show()
            plt.savefig(f'figure/participant{num + 1}_{title}_{key}')
            plt.close()




