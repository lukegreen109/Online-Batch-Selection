import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model(m_type='linear', input_dim=10, num_classes=10, in_channels=1, pretrained=False):
    model = Linear(input_dim=10, num_classes=num_classes)  # default model
    return model

class Linear(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Linear, self).__init__()
        self.fc = nn.Linear(input_dim,num_classes)

    def forward(self, x, **kwargs):
        feature = kwargs.get('need_features', False)
        feat = x
        x = self.fc(x)

        if feature:
            return x, feat
        else:
            return x
    
    def feat_nograd_forward(self, x):
        with torch.no_grad():
            # feat = torch.flatten(x, 1)
            feat = x
        x = self.fc(feat)
        return x, feat