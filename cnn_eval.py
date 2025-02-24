import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from cnn import CNN
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class CNNEvaluation():
    def __init__(self, torch_model, epoch, dataloader, fold_path):
        self.model = torch_model
        self.eval_dataloader = dataloader
        self.epoch = epoch
        self.fold_path = fold_path
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"# use the Apple Silicon GPU

    def evaluate_model(self, eval_data_loader, model):
        # Define loss function
        writer = SummaryWriter(log_dir=f'runs/{self.model}_eval')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # Initialize lists to store true labels and predictions
        y_true = []
        y_pred = []
        results = []
        for epoch in range(self.epoch):
            total_correct = 0.0
            total_samples = 0.0
            running_loss = 0.0
            for inputs, targets in eval_data_loader:
                # forward pass
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
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
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

            # calculate average loss and accuracy got the epoch
            avg_loss = running_loss / len(eval_data_loader)
            # accuracy
            accuracy = total_correct / total_samples

            # log the avg loss and acc for each epoch
            writer.add_scalar("Loss/Eval", avg_loss, epoch)
            writer.add_scalar('Accuracy/Eval', accuracy, epoch)
            # Append results to the DataFrame
            results.append({"Epoch": epoch, "Loss": avg_loss, "Accuracy": accuracy})

        # Save results to a CSV file
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{self.fold_path}/cnneval_1711_results{self.epoch}.csv', index=False)

        # Visualization
        # Calculate confusion matrix and classification report after all epochs
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix - Evaluation")
        plt.savefig(f'{self.fold_path}/cnn_eval_cm.png')
        plt.close()

        # Calculate and print additional metrics
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        overall_accuracy = accuracy_score(y_true, y_pred)
        print(f'Overall Accuracy: {overall_accuracy:.4f}')

        writer.flush()
        writer.close()

    def CNN_eval(self):
        # load model
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = CNN().to(device)
        model.load_state_dict(torch.load(f'{self.fold_path}/{self.model}.pt'))

        # set to evaluation mode
        model.eval()

        # evaluate model
        self.evaluate_model(self.eval_dataloader, model)
        print("Finished")

