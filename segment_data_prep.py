import pandas as pd
import numpy as np
import os
import re
import glob

def merged_df(matching_files):
    all_df = []
    for file_path in matching_files:
        df = pd.read_parquet(file_path)
        all_df.append(df)
    merged_df= pd.concat(all_df, ignore_index=True)
    return merged_df

# Set the base path, subfolders, and task filename
base_dir = '/home/Shared/xinyi/blob1/thesis/data/parquet_samples/'
file_pattern = os.path.join(base_dir, '**', 'radar_samples_192.168.67.112*')

matching_files = glob.glob(file_pattern, recursive=True)
split_num = 11
training_set = matching_files[:split_num]
test_set = matching_files[split_num:]



merged_df_train = merged_df(training_set)
merged_df_test = merged_df(test_set)
# save as parquet files
merged_df_train.to_parquet('/home/Shared/xinyi/blob1/thesis/radar_112/all_part_seg_train.parquet')
merged_df_test.to_parquet('/home/Shared/xinyi/blob1/thesis/radar_112/all_part_seg_test.parquet')
print('wengweng!')
