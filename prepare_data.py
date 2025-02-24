from data_preprocess import DivideData
import pandas as pd
import os
import re

"""
give the radar number, it will find the parquet data and divide by participants and required tasks
"""
dates = [23]

# radar_112: T0, 6, 7, 11
# radar_114: T8, 9, 10

pattern = r'radar_samples_192\.168\.67\.112_\d+\.parquet'
#pattern = r'radar_samples_192\.168\.67\.114_\d+\.parquet'

for date in dates:
    print(date)
    folder_path = f'./data/parquet_samples/{date}_06_22'
    all_files = os.listdir(folder_path)
    file_name = [filename for filename in all_files if filename.startswith('radar_samples')]
    for file in file_name:
        match = re.match(pattern, file)
        if match:
            #radar_num = match.group(1)
            #print(radar_num)
            #print(f'{folder_path}/{file}')
            data = pd.read_parquet(f'{folder_path}/{file}', engine='fastparquet')
            tasks = pd.read_csv('data/task.csv')

            divider = DivideData(data, tasks, f'radar_112/{date}')
            divider.divide_task_parquet([0, 6, 7, 11])
            print('wengweng!!')