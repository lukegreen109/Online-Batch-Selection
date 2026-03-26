import torch
import torch.nn as nn
import torch.nn.functional as F


def create_model(m_type='mnist_cnn', in_channels=1, num_classes=10, pretrained=False, dropout=0.25):
    # Extremely simple CNN intended for MNIST-like inputs; ignore pretrained.
    return mnist_cnn(in_channels=in_channels, num_classes=num_classes, dropout=dropout)


class mnist_cnn(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, dropout=0.25):
        super(mnist_cnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fix feature map size to be robust across small image-size variations.
        self.adapt = nn.AdaptiveAvgPool2d((7, 7))

        self.drop2d = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, **kwargs):
        need_features = kwargs.get('need_features', False)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = self.drop2d(x)
        x = self.adapt(x)
        x = torch.flatten(x, start_dim=1)

        feat = F.relu(self.fc1(x))
        feat = self.drop(feat)
        out = self.fc2(feat)

        if need_features:
            return out, feat
        else:
            return out

    def feat_nograd_forward(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = self.pool(x)

            x = F.relu(self.conv2(x))
            x = self.pool(x)

            x = self.drop2d(x)
            x = self.adapt(x)
            x = torch.flatten(x, start_dim=1)

            feat = F.relu(self.fc1(x))
            feat = self.drop(feat)

        out = self.fc2(feat)
        return out, feat
