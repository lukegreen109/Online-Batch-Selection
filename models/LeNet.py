import torch
import torch.nn as nn
import torch.nn.functional as F


def create_model(m_type='lenet', in_channels=1, num_classes=10, pretrained=False, dropout=0, **kwargs):
    model = lenet(in_channels=in_channels, num_classes=num_classes)
    return model


class lenet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(lenet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x, **kwargs):
        feature = kwargs.get('need_features', False)

        # Official PyTorch LeNet tutorial expects 32x32; MNIST is 28x28.
        if x.shape[-2:] != (32, 32):
            raise ValueError('LeNet expects 32x32 inputs.')

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        out = self.fc3(feat)

        if feature:
            return out, feat
        else:
            return out

    def feat_nograd_forward(self, x):
        with torch.no_grad():
            if x.shape[-2:] != (32, 32):
                raise ValueError('LeNet expects 32x32 inputs.')

            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = torch.flatten(x, start_dim=1)

            x = F.relu(self.fc1(x))
            feat = F.relu(self.fc2(x))

        out = self.fc3(feat)
        return out, feat