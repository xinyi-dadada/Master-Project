import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from cnn_seg import CNNSegment
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class CNNSegTrain():
    def __init__(self, epoch, model_name, dataloader, select_num):
        self.epoch = epoch
        self.model_name = model_name
        self.data_loader = dataloader
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
        plt.ylim(0,0.1)
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
        plt.savefig(f'/home/Shared/xinyi/blob1/thesis/logs_seg/{self.model_name}.png')
        plt.close()
        print('Shasha finished drawing:)')
    def confusion_matrix_visual(self, predicted, labels, num):
        predicted_np = predicted.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        # Flatten the arrays if necessary
        predicted_np = predicted_np.flatten()
        labels_np = labels_np.flatten()

        # Compute the confusion matrix
        cm = confusion_matrix(labels_np, predicted_np)

        # Plot confusion matrix using seaborn heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'/home/Shared/xinyi/blob1/thesis/logs_seg/confusion_matrix_train_{self.model_name}_{num}.png')
        plt.close()

    def train_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CNNSegment().to(device)
        #writer = SummaryWriter(log_dir=f'logs_seg/{self.model_name}_train')
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        accuracy_l = []
        loss_l = []
        # Define weights
        weight_tp = 5  # Weight for True Positives
        weight_fn = 0  # Penalty weight for False Negatives
        weight_tn = 1 # Weight for True Negatives
        weight_fp = 0  # Weight for False Positives
        for epoch in range(self.epoch):
            model.train()
            running_loss = 0.0
            for batch_data, batch_labels in self.data_loader:
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
                ## backward pass and optimization
                optimizer.zero_grad()
                # backward pass
                loss.backward()
                # optimize
                optimizer.step()
                # accumulate running loss for each batch
                running_loss += loss.item()
                # calculate the accuracy
                # 0 or 1
                # Apply a threshold to convert probabilities to binary predictions
                threshold = 0.3
                predicted = (outputs > threshold).float()
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

                # calculate accuracy
                #accuracy = (predicted == selected_labels).float().mean().item()
                accuracy_l.append(accuracy)
                # calculate average loss and accuracy for the epoch
                avg_loss = running_loss / len(selected_part_data)
                loss_l.append(avg_loss)
                print(f'loss: {avg_loss}, accuracy: {accuracy}')
                # log the average loss and accuracy for the epoch
                #writer.add_scalar('Loss/Train', avg_loss, epoch)
                #writer.add_scalar('Accuracy/Train', accuracy, epoch)
        self.confusion_matrix_visual(predicted, selected_labels, epoch)
            #writer.flush()
            #writer.close()
        torch.save(model.state_dict(), f'./{self.model_name}') # model saving also does not work:(
        self.visualize_result(accuracy_l, loss_l)
        print("Wengweng finished training the segmentation!")