import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def create_model(m_type='small_cnn', in_channels=3, num_classes=10, pretrained=False):
    model = small_cnn(in_channels=in_channels, num_classes=num_classes)  # default model
    return model

class small_cnn(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout=0):
        super(small_cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 3)
        self.conv1_drop = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv3_drop = nn.Dropout2d(p=dropout)
        if in_channels == 1: # use for mnist
            self.fc1 = nn.Linear(256 * 1 * 1, 128)
        else:
            self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc1_drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 256)
        self.fc2_drop = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, **kwargs):

        feature = kwargs.get('need_features', False)

        x = F.relu(self.pool(self.conv1_drop(self.conv1(x))))
        x = F.relu(self.pool(self.conv2_drop(self.conv2(x))))
        x = F.relu(self.pool(self.conv3_drop(self.conv3(x))))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc2_drop(self.fc2(x)))
        feat = x
        x = self.fc3(x)
        
        if feature:
            return x, feat
        else:
            return x