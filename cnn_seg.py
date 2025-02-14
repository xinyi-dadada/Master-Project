import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSegment(nn.Module):
    def __init__(self):
        super(CNNSegment, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=9, out_channels=36, kernel_size=5, padding=1, dtype=torch.float32)
        self.pool2d = nn.MaxPool2d(kernel_size=5, stride=5)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=36, out_channels=72, kernel_size=5, padding=1, dtype=torch.float32)
        self.batch_norm1 = nn.BatchNorm2d(72)
        self.dropout2 = nn.Dropout2d(p=0.25)

        self.upsample2d = nn.UpsamplingNearest2d(scale_factor=10)
        self.conv3 = nn.Conv2d(in_channels=72, out_channels=36, kernel_size=3, dtype=torch.float32)
        self.conv4 = nn.Conv2d(in_channels=36, out_channels=9, kernel_size=5, padding=1, dtype=torch.float32)
        self.dropout3 = nn.Dropout2d(p=0.25)

        self.conv_output = nn.Conv2d(in_channels=9, out_channels=2, kernel_size=10, padding=1, dtype=torch.float32)
        self.batch_norm2 = nn.BatchNorm2d(2)

        self.fc = self.fc = nn.Sequential(
            nn.Linear(25169, 12000),  # Placeholder size, to be updated dynamically
            nn.Sigmoid()
        )  # will initialize in the forward
        self.pool1d = nn.MaxPool1d(kernel_size=30, stride=30)
        self.dropout = nn.Dropout(p=0.1)
        self.dropout_fc = nn.Dropout(p=0.25)  # will initialize in the forward

    def forward(self, x):

        x = x.to(torch.float32)
        # Convolutional Layers
        x = F.relu(self.conv1(x))
        x = self.pool2d(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2d(x)
        x = self.dropout2(x)
        x = self.batch_norm1(x)

        # Upsampling and Convolutional Layers
        x = self.upsample2d(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout3(x)
        x = F.relu(self.conv_output(x))
        x = self.batch_norm2(x)

        # Flatten the spatial dimensions (height, width) into a single dimension
        batch_size = x.size(0)
        flattened_features = x.view(batch_size, -1)  # Shape: (batch_size, 9 * height * width)

        x = self.pool1d(flattened_features)
        # Dynamically create the fully connected layer based on input size

        input_size = x.size(1)  # Dynamically get the input size (9 * height * width)
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1200),  # Create fc dynamically
            nn.Sigmoid()
        ).to(x.device)


        # Pass through the fully connected layer
        x = self.fc(x)
        x = self.dropout_fc(x)
        return x




"""

class CNNSegment(nn.Module):
    def __init__(self):
        super(CNNSegment, self).__init__()
        
        # layer 1: conv2d -> maxpool -> conv2d -> maxpool
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=36, kernel_size=5, padding=1, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=20, stride=20),
            nn.Conv2d(in_channels=36, out_channels=72, kernel_size=5, padding=1, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=5)
        )

        # layer 2: upsample -> conv2d * 2 -> upsample to the original shape
        self.layer2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=10),
            nn.Conv2d(in_channels=72, out_channels=36, kernel_size=3, padding=1, dtype=torch.float32),
            nn.ReLU(),
            nn.Conv2d(in_channels=36, out_channels=9, kernel_size=3, padding=1, dtype=torch.float32),
            nn.ReLU(),
            nn.Upsample(size=(24000, 256), mode='bilinear', align_corners=False)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=72, kernel_size=10, padding=10, dtype=torch.float32),
            #nn.Conv2d(in_channels=72, out_channels=240, kernel_size=100, padding=10, dtype=torch.float32),
            nn.ReLU(),
            nn.MaxPool2d(50, 10),
        )

        # layer 4:
        self.layer4 = nn.Sequential(
            nn.Linear(34416, 24000, dtype=torch.float32),
            nn.ReLU(),
            #nn.Linear(48000, 24000, dtype=torch.float32),
            nn.Softmax(dim=1)
        )
        
        # pool
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=36, kernel_size=5, padding=1, dtype=torch.float32)
        self.pool2d = nn.MaxPool2d(kernel_size=5, stride=5)
        self.conv2 = nn.Conv2d(in_channels=36, out_channels=72, kernel_size=5, padding=1, dtype=torch.float32)
        self.batch_norm1 = nn.BatchNorm2d(72)  # current channels number=72
        # concatenate?

        # unpool or upsample
        self.conv3 = nn.Conv2d(in_channels=72, out_channels=36, kernel_size=3, dtype=torch.float32)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=5, stride=5)
        self.conv4 = nn.Conv2d(in_channels=36, out_channels=9, kernel_size=5, padding=1, dtype=torch.float32)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=10, stride=10)

        self.upsample2d = nn.UpsamplingNearest2d(scale_factor=10)
        self.upsample = nn.Upsample(size=(24000, 256), mode='bilinear', align_corners=False)
        self.batch_norm2 = nn.BatchNorm2d(9) # current channels number=9
        self.fc = None # will initialize in the forward
        self.conv_output = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=10, padding=1, dtype=torch.float32)
        self.pool1d = nn.MaxPool1d(kernel_size=30, stride=30)
        self.softmax = nn.Softmax(dim=1)
        #self.softmax = nn.Softmax2d(dim=1)
        # zero pad?


    def forward(self, x):
        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.layer4(x)



        
        x = x.to(torch.float32)
        x = nn.functional.relu(self.conv1(x))
        x = self.pool2d(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2d(x)
        #x = self.batch_norm1(x)


        #x = self.unpool2(x, indices)
        #x = self.unpool1(x, indices)
        x = self.upsample2d(x)
        x = nn.functional.relu(self.conv3(x))
        #x = self.upsample2d(x)
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv_output(x))
        #x = x.squeeze(1)
        # Flatten the spatial dimensions (height, width) into a single dimension
        batch_size = x.size(0)
        flattened_features = x.view(batch_size, -1)  # Shape: (batch_size, 9 * height * width)

        x = self.pool1d(flattened_features)
        # Dynamically create the fully connected layer based on input size
        if self.fc is None:
            input_size = x.size(1)  # Dynamically get the input size (9 * height * width)
            self.fc = nn.Sequential(
                nn.Linear(input_size, 12000).to(x.device),  # Create fc dynamically
                nn.Sigmoid()
            )

        # Pass through the fully connected layer
        x = self.fc(x)
        # softmax!!!!!
        #x = self.softmax(x)



        # Flatten the spatial dimensions (4789, 86) into a single dimension
        #x = x.view(x.size(0), -1)  # Shape: (batch_size, 9 * 4789 * 86)
        # Apply a fully connected layer to match the label size (12000)
        #x = self.fc(x)  # Shape: (batch_size, 12000)
        # Dynamically apply upsampling using the size from the label tensor size
        #x = F.interpolate(x, size=labels_shape, mode='bilinear', align_corners=False)

        #x = self.conv_output(x)

        #x = self.batch_norm2(x)

        return x
"""