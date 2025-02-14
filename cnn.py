import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolution Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=18, kernel_size=1000, dtype=torch.float32),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=30, stride=30)
        )

        # Convolution Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=18, out_channels=36, kernel_size=500, dtype=torch.float32),
            nn.BatchNorm1d(36),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=30, stride=25)
        )

        # Output Layer
        self.layer3 = nn.Sequential(
            # adaptive pooling to ensure a fixed size output
            nn.AdaptiveMaxPool1d(1),
            # fully connected layer
            nn.Linear(36, 18, dtype=torch.float32),
            nn.Dropout(p=0.5),
            nn.Linear(18, 9, dtype=torch.float32),
            # Output layer for 5 classes
            nn.Linear(9, 5, dtype=torch.float32),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

