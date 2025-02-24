import os
import glob
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.fft import fft
from data_preprocess import DivideData

class UnsupervisedClassify():

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.tasks = pd.read_csv('~/thesis/data/task.csv')

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
        # return a1_rx, a2_rx, b1_rx do need this step for adding them to the df !!!
        a1_rx_nested_arr = a1_rx.tolist()
        a2_rx_nested_arr = a2_rx.tolist()
        b1_rx_nested_arr = b1_rx.tolist()

        return a1_rx_nested_arr, a2_rx_nested_arr, b1_rx_nested_arr

    def flatten_df(self, df):
        df = df.iloc[:, :6]
        a1_rx, a2_rx, b1_rx = self.prepare_merged_channels(df)
        df['a1_rx'] = a1_rx
        df['a2_rx'] = a2_rx
        df['b1_rx'] = b1_rx
        part_arr = np.empty((9, 1), dtype=object)

        k = 0
        for col in df.columns:
            data_series = df[col]
            data_series_exploded = data_series.explode(col)
            merged_list = data_series_exploded.tolist()
            part_arr[k, 0] = merged_list
            k += 1
        df_new = pd.DataFrame(part_arr.T, columns=df.columns)
        return df_new

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
            print("cannot find task index")


class PCAandKMeans():
    def __init__(self):
        self.roi_task = ['T_0', 'T_1', 'T_2', 'T_3', 'T_4', 'T_5', 'T_6', 'T_7', 'T_8', 'T_9', 'T_10']

    def apply_fft(self, row):
        fft_result = fft(row)
        fft_magnitude = np.abs(fft_result)
        fft_magnitude_no_dc = fft_magnitude[1:len(fft_result) // 2]
        return fft_magnitude_no_dc

    def plot_PCA(self, components, mean_values):
        plt.plot(mean_values, linestyle='--', color='red', linewidth=2, label='Mean value')
        # Add labels for each of the first three components
        component_labels = ['First Component', 'Second Component', 'Third Component']
        for i in range(3):
            plt.plot(components[:, i], alpha=0.5, label=component_labels[i])
        plt.grid(False)
        plt.title(f"PCA Components and Mean Value for Participant {participant_num}")
        plt.xlabel("Observation Index")
        plt.ylabel("Component Value")
        plt.savefig(f'~/thesis/figure/result_pca/participant_{participant_num}.jpg')
        plt.close()
    def apply_PCA(self, df_new, n_tasks):
        fft_for_pca = df_new['fft_result']
        min_length = min(fft_for_pca.apply(len))
        fft_for_pca = np.array(fft_for_pca)
        truncated_arrays = np.array([arr[:min_length] for arr in fft_for_pca])
        pca = PCA(n_components=11)
        components = pca.fit_transform(truncated_arrays)
        mean_values = np.mean(components[:, :3], axis=1)
        top_indices = np.argsort(mean_values)[-n_tasks:][::-1]  # Get the last four indices and reverse them for descending order

        # Print the indices of the top four tasks
        print("Indices of the top three tasks:", top_indices)
        top_tasks_df = df_new[df_new['task'].isin(top_indices)]
        #self.plot_PCA(components, mean_values, participant_num)
        return top_tasks_df, components, mean_values

    def fft_df(self, part_1, n_tasks):
        part_1 = part_1.reset_index(drop=True)
        task_index = unsupervied_classify.divide_task_index(part_1)
        col = 'rx1_freq_a_channel_i_data'
        task_arr = np.empty((11, 1), dtype=object)
        try:
            for key, value in task_index.items():

                if key in self.roi_task:  # key: T_0, T_1, ...
                    start, end = value
                    task_num = int(key.split('_')[1])
                    data_series = part_1[col]
                    data_series_exploded = data_series.explode(col)
                    merged_list = data_series_exploded.tolist()
                    task_l = merged_list[start:(end + 1)]
                    task_arr[task_num, 0] = task_l
        except Exception as e:
            print("Something went Wrong @_@")

        if any(task_arr[:, 0]):
            df_new = pd.DataFrame(task_arr)
            df_new['task'] = df_new.index
            df_new.rename(columns={df_new.columns[0]: 'rx1_freq_a_channel_i_data'}, inplace=True)
            df_new['fft_result'] = df_new['rx1_freq_a_channel_i_data'].apply(self.apply_fft)
            top_tasks_df, components, mean_values = self.apply_PCA(df_new, n_tasks)
            return top_tasks_df, components[:, :3], mean_values

    def Kmeans_result(self, df_new, n_cluster):
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        fft_for_pca = df_new['fft_result']
        min_length = min(fft_for_pca.apply(len))
        fft_for_pca = np.array(fft_for_pca)
        truncated_arrays = np.array([arr[:min_length] for arr in fft_for_pca])
        kmeans.fit(truncated_arrays)
        labels = kmeans.labels_
        df_new[f'clusters_{n_cluster}'] = labels

        # statistics
        cluster_task_counts = df_new.groupby(f'clusters_{n_cluster}')['task'].nunique().reset_index()
        cluster_task_counts.columns = ['Cluster', 'Number of Unique Tasks']
        # Count occurrences of each task in each cluster
        task_cluster_counts = df_new.groupby([f'clusters_{n_cluster}', 'task']).size().unstack(fill_value=0)

        # Plotting
        # Define color mapping
        color_mapping = {0: 'blue', 6: 'pink', 7: 'green'}
        default_color = 'gray'

        # Generate color list based on task columns
        colors = [color_mapping.get(task, default_color) for task in task_cluster_counts.columns]

        plt.figure(figsize=(12, 8))
        task_cluster_counts.plot(kind='bar', stacked=True, color=colors, ax=plt.gca())
        plt.title(f'Task Distribution Across K-Means cluster = {n_cluster}')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Tasks')
        plt.xticks(rotation=0)  # Rotate x labels for better readability

        # Custom legend setup
        unique_tasks = [0, 6, 7]  # Specify unique tasks to show individually
        legend_labels = ["Task 0", "Task 6", "Task 7", "Other Tasks"]
        handles = [plt.Line2D([0], [0], color=color_mapping.get(task, default_color), linewidth=10) for task in
                   unique_tasks]
        handles.append(plt.Line2D([0], [0], color=default_color, linewidth=10))  # Add single "Other Tasks" entry

        plt.legend(handles=handles, labels=legend_labels, title="Task", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'~/thesis/figure/result_kmeans/cluster_{n_cluster}.jpg')
        plt.close()

pca_kmeans = PCAandKMeans()
radar_no = ...

# Set the base path, subfolders, and task filename
base_dir = '~/thesis/data/parquet_samples/'
unsupervied_classify = UnsupervisedClassify(base_dir)

# Define the file pattern to search for radar files
file_pattern = os.path.join(base_dir, '**', f'radar_samples_192.168.67.{radar_no}*')
matching_files = glob.glob(file_pattern, recursive=True)

# Initialize an empty list to collect results
all_participants_results = []
pca_results = []
participant_num = 0
# Iterate through each matching file
for data_path in matching_files:
    divide_data = DivideData(data_path)
    part, num = divide_data.divide_participants()
    for key, value in enumerate(part):
        participant_num += 1
        data_part = part[value]
        try:
            pca_results, components, mean_values = pca_kmeans.fft_df(data_part, 3)
        except:
            continue
        components = np.array(components)
        mean_values = np.array(mean_values)
        result_entry = {
            'participant_num': participant_num,
            'components': components.tolist(),  # Store as list to create nested arrays
            'mean_values': mean_values.tolist()  # Convert to list for easier storage
        }
        pca_results._append(result_entry, ignore_index=True)
        #all_participants_results.append(pca_results)
pca_results = pd.DataFrame(pca_results)
# concat all the chosen tasks and clustering by kmeans
#df_all_participants = pd.concat(all_participants_results, ignore_index=True)
df_pca_results = pd.concat(pca_results, ignore_index=True)
df_pca_results.to_parquet('/home/Shared/xinyi/blob1/thesis/pca_result.parquet')
#pca_kmeans.Kmeans_result(df_new=df_all_participants, n_cluster=5)
