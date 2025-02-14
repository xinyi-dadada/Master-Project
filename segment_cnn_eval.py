import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from cnn_seg import CNNSegment
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class CNNEvaluation():
    def __init__(self, torch_model, epoch, dataloader, select_num):
        self.model = torch_model
        self.dataloader = dataloader
        self.epoch = epoch
        self.select_num = select_num

    def visualize_result(self, accuracy_history, loss_history):
        # Assume loss_history and accuracy_history are lists that store loss and accuracy for each epoch
        epochs = range(1, len(loss_history) + 1)

        sns.set(style="whitegrid")
        sns.set_palette("bright")
        # Plot Loss
        plt.figure(figsize=(24, 16))
        plt.subplot(2, 1, 1)
        sns.lineplot(x=epochs, y=loss_history, color='r', label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0,1)
        plt.tight_layout()
        plt.legend()

        # Plot Accuracy
        plt.subplot(2, 1, 2)
        sns.lineplot(x=epochs, y=accuracy_history, color='b', label='Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0,1)
        plt.tight_layout()
        plt.legend()

        plt.savefig(f'/home/Shared/xinyi/blob1/thesis/logs_seg/test_{self.model}.png')
        plt.close()
        print('Shasha finished drawing again :)')

    def confusion_matrix_visual(self, cm, num):

        # Plot confusion matrix using seaborn heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'/home/Shared/xinyi/blob1/thesis/logs_seg/confusion_matrix_eval_{self.model}_{num}.png')
        plt.close()

    def evaluate_model(self, eval_data_loader, model):
        # Define loss function (e.g., CrossEntropyLoss for classification tasks)
        # MSELoss: measure the mean square error
        #criterion = nn.MSELoss()
        #optimizer = optim.SGD(model.parameters(), lr=0.1)
        #writer = SummaryWriter(log_dir=f'runs/{self.model}_eval')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        accuracy_l = []
        loss_l = []
        # Define weights
        weight_tp = 5  # Weight for True Positives
        weight_fn = 0  # Penalty weight for False Negatives
        weight_tn = 1  # Weight for True Negatives
        weight_fp = 0  # Weight for False Positives
        for epoch in range(self.epoch):
            running_loss = 0.0
            for batch_data, batch_labels in eval_data_loader:
                # randomly choose some participants from the data loader
                part_num = batch_data.shape[0]
                selected_indices = random.sample(range(part_num), self.select_num)
                selected_part_data = batch_data[selected_indices, :, :, :]
                selected_labels = batch_labels[selected_indices]
                selected_labels = selected_labels.to(torch.float32)
                # to device
                selected_part_data = selected_part_data.to(device)
                selected_labels = selected_labels.to(device)

                # forward pass
                outputs = model(selected_part_data)

                # calculate loss
                loss = criterion(outputs, selected_labels)

                # backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # accumulate running loss for each batch
                running_loss += loss.item()

                # calculate the accuracy

                predicted = (outputs > 0.5).float()  # Convert to 0 or 1
                predicted_np = predicted.cpu().detach().numpy()
                labels_np = selected_labels.cpu().detach().numpy()
                # Flatten the arrays if necessary
                predicted_np = predicted_np.flatten()
                labels_np = labels_np.flatten()
                cm = confusion_matrix(labels_np, predicted_np, labels=[0, 1])
                # Extract counts
                tn, fp, fn, tp = cm.ravel()
                # Calculate weighted accuracy
                weighted_score = (weight_tp * tp +
                                  weight_fn * fn +
                                  weight_tn * tn +
                                  weight_fp * fp)
                # Normalize by the number of samples to get average score
                total_samples = tn + fp + fn + tp
                accuracy = weighted_score / total_samples

                # calculate average loss and accuracy got the epoch
                avg_loss = running_loss / len(eval_data_loader)
                loss_l.append(avg_loss)
                # accuracy
                accuracy_l.append(accuracy)
                print(f'evaluation accuracy: {accuracy}, loss: {avg_loss}')
        self.confusion_matrix_visual(cm, epoch)
        self.visualize_result(accuracy_l, loss_l)
        print('Shasha hua ya hua')


    def CNNSegEval(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load model
        model = CNNSegment()
        model.to(device)
        model.load_state_dict(torch.load(f'/home/Shared/xinyi/blob1/thesis/logs_seg/{self.model}.pt', map_location=device))
        # set to evaluation mode
        model.eval()

        #data_tensor = torch.tensor(self.eval_data, dtype=torch.float64)
        #labels_tensor = self.eval_labels
        #labels_tensor = torch.tensor(self.eval_labels, dtype=torch.float64)
        #dataset = TensorDataset(data_tensor, labels_tensor)
        #eval_data_loader = DataLoader(dataset, batch_size=32)

        # evaluate model
        self.evaluate_model(self.dataloader, model)
        print('Wengweng finished evaluating the model ^^')



