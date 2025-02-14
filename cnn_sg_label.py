import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, padding=1, dtype=torch.float64)
        self.pool1 = nn.MaxPool1d(kernel_size=20, stride=20)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=1, dtype=torch.float64)
        self.pool2 = nn.MaxPool1d(kernel_size=25, stride=25)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=1, dtype=torch.float64)
        self.pool3 = nn.MaxPool1d(kernel_size=25, stride=25)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=1, dtype=torch.float64)
        self.pool4 = nn.MaxPool1d(kernel_size=25, stride=25)
        #self.pool4 = nn.MaxPool1d(kernel_size=18, stride=18)
        self.fc1 = nn.Linear(256, 128, dtype=torch.float64)
        #self.fc1 = nn.Linear(128, 64, dtype=torch.float64)
        self.fc2 = nn.Linear(128, 7, dtype=torch.float64)
        #self.fc2 = nn.Linear(64, 7, dtype=torch.float64)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool3(x)
        x = nn.functional.relu(self.conv4(x))
        x = self.pool4(x)
        x = torch.flatten(x, 1)

        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

### prepare the data for training model and evaluating ###
class CNNDataPrepare():
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def merge_and_pad(self, arrays, target_shape):
        # merge the np arr for each task together, padding to make each arr with the same length
        # Create a new array filled with a padding value
        padded_array = np.zeros(target_shape, dtype=arrays[0].dtype)  # Assuming all arrays have the same dtype

        # Copy elements from each array into the new array, with padding
        for i, arr in enumerate(arrays):
            padded_array[i, :, :arr.shape[1]] += arr

        return padded_array

    def data_prep(self):
        # load the parquet dataset
        data = pd.read_parquet(self.path)

        # split training and test set
        training_set, test_set = train_test_split(data, test_size=0.33, random_state=42)
        # reset index
        training_set.reset_index(drop=True, inplace=True)
        test_set.reset_index(drop=True, inplace=True)

        # column names for grouped set
        col_names = ['rx1_freq_a_channel_i_data', 'rx1_freq_a_channel_q_data',
                     'rx2_freq_a_channel_i_data', 'rx2_freq_a_channel_q_data',
                     'rx1_freq_b_channel_i_data', 'rx1_freq_b_channel_q_data']

        # merge the signal with same task together
        # group by the task column and concatenate the values of the first 6 columns
        grouped_training = training_set.groupby('Task').agg(lambda x: x.tolist())
        # create a new df with six channels and task, shape(7,6)
        training_df = pd.DataFrame(grouped_training[col_names].values.tolist(), columns=col_names)
        training_df['Task'] = grouped_training.index

        # same for test set
        grouped_test = test_set.groupby('Task').agg(lambda x: x.tolist())
        # create a new df with six channels and task, shape(7,6)
        test_df = pd.DataFrame(grouped_test[col_names].values.tolist(), columns=col_names)
        test_df['Task'] = grouped_test.index

        # get data and labels
        training_data, training_labels = training_df.iloc[:, :6], training_df['Task']
        test_data, test_labels = test_df.iloc[:, :6], test_df['Task']


        # convert data to list for array
        training_task = training_data.values.tolist()
        test_task = test_data.values.tolist()
        # padding and merge all task array together
        training_task_arrays = [np.array(training_task[i]) for i in range(7)]
        test_task_arrays = [np.array(test_task[i]) for i in range(7)]
        #print(f'training set shape: {training_task_arrays[4].shape}\n'
        #      f'test set shape: {test_task_arrays[4].shape}')
        # set the target shape with 638976 after checking every task array, could be 6400000/480000 for all the data... good number^^
        training_target_shape = (7, 6, 900000)
        test_target_shape = (7, 6, 900000)

        # Merge and pad the arrays
        training_merged_array = self.merge_and_pad(training_task_arrays, training_target_shape)
        test_merged_array = self.merge_and_pad(test_task_arrays, test_target_shape)

        print('weng~')
        return training_merged_array, test_merged_array

class CNNTrain():
    def __init__(self, train_name, epoch, model_name, data, labels):
        self.train_name = train_name
        self.epoch = epoch
        self.model_name = model_name
        # self.data = np.load(f'cnn_data/{train_name}_task_training.npy', allow_pickle=True)
        # self.data = np.load(f'cnn_data/radar_112_task_training.npy', allow_pickle=True)
        # self.labels = np.load(f'cnn_data/labels_06.npy', allow_pickle=True)
        self.data = data
        self.labels = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)

    def train_model(self, data_loader):
        model = CNN()
        # model.to(torch.device('cuda:0'))
        # criterion = nn.MSELoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.1)
        writer = SummaryWriter(log_dir=f'runs/{self.model_name}_train')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Choose an optimizer

        for epoch in range(self.epoch):
            total_correct = 0.0
            total_samples = 0.0
            running_loss = 0.0
            for batch_data, batch_labels in data_loader:
                # forward pass
                outputs = model(batch_data)
                outputs = outputs.to(torch.float32)
                # calculate loss
                loss = criterion(outputs, batch_labels)
                # backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # accumulate running loss for each batch
                running_loss += loss.item()

                # calculate the accuracy
                # get the index of the max log-probability
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

            # calculate average loss and accuracy for the epoch
            avg_loss = running_loss / len(data_loader)
            # calculate accuracy
            accuracy = total_correct / total_samples

            # log the average loss and accuracy for the epoch
            writer.add_scalar('Loss/Train', avg_loss, epoch)
            writer.add_scalar('Accuracy/Train', accuracy, epoch)

        writer.flush()
        writer.close()

        torch.save(model.state_dict(), f'{self.model_name}.pt')

    def train(self):
        """
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(self.labels)

        # perform one-hot encoding, should it be done during dataset preparation
        num_classes = len(label_encoder.classes_)
        labels_one_hot = np.zeros((len(labels_encoded), num_classes))
        labels_one_hot[np.arange(len(labels_encoded)), labels_encoded] = 1
        """

        data_tensor = torch.tensor(self.data, dtype=torch.float64)
        labels_tensor = self.labels
        # labels_tensor = torch.tensor(self.labels, dtype=torch.float64)
        dataset = TensorDataset(data_tensor, labels_tensor)
        data_loader = DataLoader(dataset, batch_size=32)
        self.train_model(data_loader)
        print('wengweng~~')

class CNNEvaluation():
    def __init__(self, torch_model, eval_name, epoch, data, labels):
        self.model = torch_model
        self.eval_data = data
        self.eval_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int64)
        self.epoch = epoch

    # tensorboard
    def evaluate_model(self, eval_data_loader, model):
        # Define loss function (e.g., CrossEntropyLoss for classification tasks)
        # MSELoss: measure the mean square error
        #criterion = nn.MSELoss()
        #optimizer = optim.SGD(model.parameters(), lr=0.1)
        writer = SummaryWriter(log_dir=f'runs/{self.model}_eval')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Choose an optimizer
        for epoch in range(self.epoch):
            total_correct = 0.0
            total_samples = 0.0
            running_loss = 0.0
            for inputs, targets in eval_data_loader:
                # forward pass
                outputs = model(inputs)
                outputs = outputs.to(torch.float32)
                # calculate loss
                loss = criterion(outputs, targets)
                # accumulate running loss for each batch
                running_loss += loss
                # backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # accumulate running loss for each batch
                running_loss += loss.item()

                # calculate the accuracy
                # get index of the max log-probability
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

            # calculate average loss and accuracy got the epoch
            avg_loss = running_loss / len(eval_data_loader)
            # accuracy
            accuracy = total_correct / total_samples

            # log the avg loss and acc for each epoch
            writer.add_scalar("Loss/Eval", avg_loss, epoch)
            writer.add_scalar('Accuracy/Eval', accuracy, epoch)
        writer.flush()
        writer.close()

    def CNN_eval(self):
        # load model
        model = CNN()
        #model.load_state_dict(torch.load(f'{self.model}.pt'))
        #model.to(torch.device('cuda:2'))
        # set to evaluation mode
        model.eval()

        data_tensor = torch.tensor(self.eval_data, dtype=torch.float64)
        labels_tensor = self.eval_labels
        #labels_tensor = torch.tensor(self.eval_labels, dtype=torch.float64)
        dataset = TensorDataset(data_tensor, labels_tensor)
        eval_data_loader = DataLoader(dataset, batch_size=32)

        # evaluate model
        self.evaluate_model(eval_data_loader, model)
        print('wengwengweng~~~')


path = 'radar_114.parquet'
name = 'radar_114'
model_name = f'model_2805_{name}_4'
epoch = 20

#data = np.load(f'cnn_data/radar_112_task_training.npy', allow_pickle=True)
labels = np.load('cnn_data/labels_06.npy')
#eval_data = np.load('cnn_data/radar_112_task_test.npy', allow_pickle=True)

###############################

cnn_prep = CNNDataPrepare(path=path, name=name)
train_set, test_set = cnn_prep.data_prep()
training = CNNTrain(train_name=name, epoch=epoch, model_name=model_name, data=train_set, labels=labels)
training.train()
evaluation = CNNEvaluation(torch_model=model_name, eval_name=name, epoch=epoch, data=test_set, labels=labels)
evaluation.CNN_eval()


