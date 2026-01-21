import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model(m_type='small_linear', num_classes=10, im_size=[28,28], in_channels=1, pretrained=False):
    # Simple MNIST MLP; ignore pretrained
    return small_linear(num_classes=num_classes, input_size=im_size[0]*im_size[1]*in_channels)

class small_linear(nn.Module):
    def __init__(self, input_size=3072, hidden_size=128, num_classes=10):
        super(small_linear, self).__init__()
        # First layer (feature extractor)
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Final classifier exposed as `fc` to match ResNet-style code
        self.fc = nn.Linear(hidden_size, num_classes)
        # Optional alias if existing code refers to fc2
        self.fc2 = self.fc
        # Keep an avgpool for interface compatibility (no-op)
        self.avgpool = nn.Identity()

    def forward(self, x, **kwargs):
        need_features = kwargs.get('need_features', False)
        # Flatten NHWC/NCHW image tensors to (N, 784)
        x = torch.flatten(x, 1)
        feat = F.relu(self.fc1(x))
        out = self.fc(feat)
        if need_features:
            return out, feat
        else:
            return out

    def feat_nograd_forward(self, x):
        with torch.no_grad():
            x = torch.flatten(x, 1)
            feat = F.relu(self.fc1(x))
        out = self.fc(feat)
        return out, feat