import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, side_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Adjust the size calculation based on the number of convolutional layers
        self.fc1 = nn.Linear(64 * (side_size - 4) ** 2, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 1)

        # Optionally use dropout
        self.dropout1 = nn.Dropout(0.01)
        self.dropout2 = nn.Dropout(0.01)

        print("CNN model created")

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = self.dropout1(x)  # Dropout applied after flattening

        x = F.relu(self.fc1(x))
        x = self.dropout2(x)  # Dropout applied after first fully connected layer
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(
            x
        )  # No softmax here, as CrossEntropyLoss applies LogSoftmax internally
