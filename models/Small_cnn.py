import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def create_model(m_type='Small_cnn', num_classes=10, in_channels = 1, pretrained = False):
    model = Small_cnn(in_channels = in_channels, num_classes = num_classes)
def create_model(m_type='Small_cnn', num_classes=10, in_channels = 1, pretrained = False):
    model = Small_cnn(in_channels = in_channels, num_classes = num_classes)
    return model

class Small_cnn(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, dropout=0):
        super().__init__()
class Small_cnn(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, dropout=0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 3)
        self.conv1_drop = nn.Dropout2d(p=dropout)


        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2_drop = nn.Dropout2d(p=dropout)


        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv3_drop = nn.Dropout2d(p=dropout)

        self.pool = nn.MaxPool2d(2, 2)

        # Correct feature size for MNIST
        self.fc1 = nn.Linear(256 * 1 * 1, 128)
        self.fc1_drop = nn.Dropout(dropout)


        self.fc2 = nn.Linear(128, 256)
        self.fc2_drop = nn.Dropout(dropout)


        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1_drop(self.conv1(x))))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        x = F.relu(self.pool(self.conv3_drop(self.conv3(x))))

        # ✅ batch-safe flatten
        x = torch.flatten(x, start_dim=1)
        # ✅ batch-safe flatten
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        x = self.fc3(x)
        return x


# class Small_cnn(nn.Module):
#     def __init__(self, in_channels=3, num_classes=10, dropout=0):
#         super(Small_cnn, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, 64, 3)
#         self.conv1_drop = nn.Dropout2d(p=dropout)
#         self.conv2 = nn.Conv2d(64, 128, 3)
#         self.conv2_drop = nn.Dropout2d(p=dropout)
#         self.conv3 = nn.Conv2d(128, 256, 3)
#         self.conv3_drop = nn.Dropout2d(p=dropout)
#         self.fc1 = nn.Linear(64 * 4 * 4, 128)
#         self.fc1_drop = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc2_drop = nn.Dropout(dropout)
#         self.fc3 = nn.Linear(256, num_classes)

#         self.pool = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         x = F.relu(self.pool(self.conv1_drop(self.conv1(x))))
#         x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
#         x = F.relu(self.pool(self.conv3_drop(self.conv3(x))))

#         x = x.view(-1, 64 * 4 * 4)

#         x = F.relu(self.fc1_drop(self.fc1(x)))
#         x = F.relu(self.fc2_drop(self.fc2(x)))
#         x = self.fc3(x)
#         return x


# class Small_cnn(nn.Module):
#     def __init__(self, in_channels=3, num_classes=10, dropout=0):
#         super(Small_cnn, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, 64, 3)
#         self.conv1_drop = nn.Dropout2d(p=dropout)
#         self.conv2 = nn.Conv2d(64, 128, 3)
#         self.conv2_drop = nn.Dropout2d(p=dropout)
#         self.conv3 = nn.Conv2d(128, 256, 3)
#         self.conv3_drop = nn.Dropout2d(p=dropout)
#         self.fc1 = nn.Linear(64 * 4 * 4, 128)
#         self.fc1_drop = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(128, 256)
#         self.fc2_drop = nn.Dropout(dropout)
#         self.fc3 = nn.Linear(256, num_classes)

#         self.pool = nn.MaxPool2d(2, 2)

#     def forward(self, x):
#         x = F.relu(self.pool(self.conv1_drop(self.conv1(x))))
#         x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
#         x = F.relu(self.pool(self.conv3_drop(self.conv3(x))))

#         x = x.view(-1, 64 * 4 * 4)

#         x = F.relu(self.fc1_drop(self.fc1(x)))
#         x = F.relu(self.fc2_drop(self.fc2(x)))
#         x = self.fc3(x)
#         return x