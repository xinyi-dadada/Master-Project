import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

"""
# Assuming input shape (batch_size, sequence_length, num_channels)
input_shape = (64, 32, 1)
batch_size, sequence_length, num_channels = input_shape

# Define MaxPool1d operation
pool = nn.MaxPool1d(kernel_size=2, stride=2)

# Create random input tensor
input_tensor = torch.randn(batch_size, sequence_length, num_channels)

# Apply MaxPool1d
output_tensor = pool(input_tensor)

# Output shape after MaxPool1d
print("Output shape:", output_tensor.shape)

"""
# pool of size=3, stride=2
m = nn.MaxPool1d(3, stride=2)
input = torch.randn(20, 16, 50)
output = m(input)
output_2 = m(output)
print(output.shape)
"""
data = pd.read_parquet('13_06_22/radar_114/radar_114_data.parquet')

data_n = data.iloc[:, :7]
training_set, test_set = train_test_split(data_n, test_size=0.33, random_state=28)
print(training_set.shape, test_set.shape)



radar_data = pd.read_parquet('13_06_22/radar_114/radar_114_data.parquet')
channels_array = radar_data.iloc[:, :6].to_numpy()
channels_array = channels_array.reshape((3423232, 6, -1))
conv_layer = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
channels_tensor = torch.tensor(channels_array, dtype=torch.float32)
output_tensor = conv_layer(channels_tensor)

"""