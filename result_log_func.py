import pandas as pd
data = pd.read_parquet('/home/Shared/xinyi/blob1/thesis/logs_seg/radar112_seg_all_2110.parquet')
bool_re = data['label'].apply(lambda x: x == [6, 7])
count = (bool_re).sum()
print(f'The number of splits including task [6, 7] is {count}')
