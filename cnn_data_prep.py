import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import KFold #k-fold cross validation
### This code is used to prepare the data for training model and evaluating ###

class NestedArrayDataset(Dataset):
    def __init__(self, data_list, labels):
        self.data = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the list of arrays to a tensor
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample, label
class CNNDataPrepare():
    def __init__(self, path, name):
        self.path = path
        self.name = name

    # Function to pad arrays to the maximum length
    def pad_array(self, arr, max_length):
        padded = np.zeros(max_length)
        padded[:len(arr)] = arr
        return padded

    # Check the DataLoader
    def check_dataloader(self, dataloader):
        for i, (data_batch, labels_batch) in enumerate(dataloader):
            print(f"Batch {i} data shape: {data_batch.shape}")
            print(f"Batch {i} labels shape: {labels_batch.shape}")
            print(f"Batch {i} data sample: {data_batch[0]}")
            print(f"Batch {i} labels sample: {labels_batch[0]}")
    def prepare_tensor(self, training_data, training_labels):
        # column names for grouped set
        col_names = ['rx1_freq_a_channel_i_data', 'rx1_freq_a_channel_q_data',
                     'rx2_freq_a_channel_i_data', 'rx2_freq_a_channel_q_data',
                     'rx1_freq_b_channel_i_data', 'rx1_freq_b_channel_q_data']

        ### prepare tensor file for cnn training and evaluation
        # need to pad all the list to the same length -> find the max length
        max_length = max(max(len(arr) for arr in training_data[col]) for col in training_data.columns)
        # pad each array in the df columns
        train_padded_data = {col: [self.pad_array(arr, max_length) for arr in training_data[col]] for col in col_names}
        # Convert padded DataFrame columns to a single NumPy array
        train_data_np = np.stack([np.vstack(train_padded_data[col]) for col in col_names], axis=1)
        # Convert NumPy array to PyTorch tensor
        train_data_tensor = torch.tensor(train_data_np, dtype=torch.float32)
        train_labels_tensor = torch.tensor(training_labels, dtype=torch.long)
        # Create the TensorDataset
        train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
        # Set the batch size
        train_batch_size = train_data_tensor.size()[0]
        # Create a DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        self.check_dataloader(train_dataloader)
        return train_dataloader

    def data_prep(self, num_folds=5):
        # load the parquet dataset
        data = pd.read_parquet(self.path)
        # Extract data and labels
        dataset = data.iloc[:, :6]
        labels = data['task']

        # Pad arrays and prepare tensors
        max_length = max(max(len(arr) for arr in dataset[col]) for col in dataset.columns)
        padded_data = {col: [self.pad_array(arr, max_length) for arr in dataset[col]] for col in dataset.columns}
        data_np = np.stack([np.vstack(padded_data[col]) for col in dataset.columns], axis=1)
        data_tensor = torch.tensor(data_np, dtype=torch.float32)
        labels_tensor = torch.tensor(labels.values, dtype=torch.long)
        # Prepare for k-fold cross-validation
        dataset = TensorDataset(data_tensor, labels_tensor)
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        folds = []
        for train_idx, val_idx in kfold.split(dataset):
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
            folds.append((train_loader, val_loader))

        return folds


