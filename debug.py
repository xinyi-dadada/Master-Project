import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
#from data import DivideData

# Assume df is your DataFrame

# Example DataFrame
data = {'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Normalizing the data
normalized_df = (df - df.mean()) / df.std()

print("Original DataFrame:")
print(df)

print("\nNormalized DataFrame:")
print(normalized_df)

"""
def plot_figure_task(df, title):
    df = np.array(df)
    df = df.flatten()

    plt.figure(figsize=(36, 6))
    plt.plot(df)
    plt.title(title)
    plt.savefig(f'participant/part_1/figure_{title}.jpg')  # need to update the
    plt.close()

df = pd.read_csv('participant/part_1/task_0.csv')
#data = pd.read_parquet("data/parquet_samples/radar_samples_192.168.67.111_409.parquet",
 #                       engine='fastparquet')
#tasks = pd.read_csv('data/task.csv')

#divider = DivideData(data, tasks)


# the signal column names
col_names = ['rx1_freq_a_channel_i_data', 'rx1_freq_a_channel_q_data',
             'rx2_freq_a_channel_i_data', 'rx2_freq_a_channel_q_data',
             'rx1_freq_b_channel_i_data', 'rx1_freq_b_channel_q_data']


data_series = pd.DataFrame(df)
data_series_exploded = data_series.explode(title)
merged_list = data_series_exploded[title].tolist()

df_task = df['rx1_freq_a_channel_i_data']
df_task = np.array(df_task)
df_task = df_task.flatten()
df_list = df_task.tolist()
print(df_list)
plt.plot(df_list)
plt.show()




for name in col_names:
    df_task = df[name]
    plot_figure_task(df_task, name)

# Plot all 6 channels in one plot
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

plt.xlabel('Index')  # Assuming index is your x-axis
plt.ylabel('Values')  # Label for y-axis
plt.title('Plot of all 6 channels')  # Title for the plot
plt.legend()  # Add legend to the plot
plt.grid(True)  # Add grid
plt.savefig(f'participant/part_1/figure_i.jpg')
plt.close()


df_channel = df['rx1_freq_a_channel_i_data'].values.tolist()
#plt.figure(figsize=(36, 6))
plt.plot(df_channel)
#plt.show()
plt.savefig(f'participant/part_1/figure_i.jpg')
plt.close()


# for every col, plot a figure to describe
for name in col_names:
    df = task_data[name]
    data_series_exploded = df.explode(name)
    merged_list = data_series_exploded[name].tolist()
    plt.figure(figsize=(36, 6))
    plt.plot(merged_list)
    plt.savefig(f'participant/part_1/figure_{name}.jpg')
    plt.close()
"""