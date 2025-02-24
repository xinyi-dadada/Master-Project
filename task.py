import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from data_preprocess import DivideData

path='13_06_22/radar_114/radar_114_data.parquet'
data = pd.read_parquet(path)
print(data.shape)


"""
# load task, data and divider
tasks = pd.read_csv('data/task.csv')
data = pd.read_parquet("data/parquet_samples/radar_samples_192.168.67.53_414.parquet",
                        engine='fastparquet')
divider = DivideData(data, tasks)

# divide participant by time, there are four participants
participants, num = divider.divide_participants()
divider.divide_task_csv(participants)



file_path = 'participant/part_1'
data = np.load(f'{file_path}/task_0.npy')


# Plot each row
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(6):
    ax.plot(data[i], label=f'Row {i+1}')

ax.legend()
plt.savefig(f'{file_path}/task_0')
plt.close()
"""