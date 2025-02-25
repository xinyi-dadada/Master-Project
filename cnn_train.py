import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from cnn import CNN
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class CNNTrain():
    def __init__(self, train_name, epoch, model_name, fold_path, data_loader):
        self.train_name = train_name
        self.epoch = epoch
        self.model_name = model_name
        self.fold_path = fold_path
        self.dataloader = data_loader
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"# use the Apple Silicon GPU

    def train_model(self, data_loader):
        model = CNN().to(self.device)
        writer = SummaryWriter(log_dir=f'runs/{self.model_name}_train')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Initialize lists to store true labels and predictions
        y_true = []
        y_pred = []
        # DataFrame to store accuracy and loss for each epoch
        results = []
        for epoch in range(self.epoch):
            total_correct = 0.0
            total_samples = 0.0
            running_loss = 0.0
            for batch_data, batch_labels in data_loader:
                # forward pass
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
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
                y_true.extend(batch_labels.cpu().numpy())  # Convert targets to CPU and add to y_true list
                y_pred.extend(predicted.cpu().numpy())  # Convert predictions to CPU and add to y_pred list
            # calculate average loss and accuracy for the epoch
            avg_loss = running_loss / len(data_loader)
            # calculate accuracy
            accuracy = total_correct / total_samples
            # log the average loss and accuracy for the epoch
            writer.add_scalar('Loss/Train', avg_loss, epoch)
            writer.add_scalar('Accuracy/Train', accuracy, epoch)
            # Append results to the DataFrame
            results.append({"Epoch": epoch, "Loss": avg_loss, "Accuracy": accuracy})

        # Save results to a CSV file
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{self.fold_path}/cnntrain_results_{self.epoch}.csv', index=False)
        # Calculate confusion matrix and classification report after all epochs
        cm = confusion_matrix(y_true, y_pred)

        # Visualization
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix - Train")
        plt.savefig(f'{self.fold_path}/cnn_train_cm.png')
        plt.close()
        # Calculate and print additional metrics
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        overall_accuracy = accuracy_score(y_true, y_pred)
        print(f'Overall Accuracy: {overall_accuracy:.4f}')
        writer.flush()
        writer.close()

        torch.save(model.state_dict(), f'{self.fold_path}/{self.model_name}.pt')

    def train(self):
        self.train_model(self.dataloader)
        print("Finished training")
