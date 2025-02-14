# merge all the participants in one radar together
import os
import re
import pandas as pd
import numpy as np
import glob

# Set the base path, subfolders, and task filename
base_dir = 'data/parquet_samples/'
radar_no = 112
file_pattern = os.path.join(base_dir, '**', f'radar_samples_192.168.67.{radar_no}*')
# get all file names for this radar
matching_files = glob.glob(file_pattern, recursive=True)

# contact all the files together and save
all_df = []
for file_path in matching_files:
    df = pd.read_parquet(file_path)
    all_df.append(df)
merged_df = pd.concat(all_df, ignore_index=True)
path_to_save = 'radar_112/all_part_seg.parquet'
merged_df.to_parquet(path_to_save)