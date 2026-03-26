import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model(m_type='simple_fc', in_channels = 1, num_classes=10, pretrained=False):
    model = Simple_fc(num_classes=num_classes)  # default model
    return model

class Simple_fc(nn.Module):
    def __init__(self, num_classes=2):
        super(Simple_fc, self).__init__()

        self.linear1 = nn.Linear(2, 16)
        self.linear2 = nn.Linear(16,16)
        self.fc = nn.Linear(16,num_classes)

    def forward(self, x, **kwargs):
        feature = kwargs.get('need_features', False)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        feat = x
        x = self.fc(x)

        if feature:
            return x, feat
        else:
            return x
    
    def feat_nograd_forward(self, x):
        with torch.no_grad():
            x = F.relu(self.linear1(x))
            x = F.relu(self.linear2(x))
            feat = torch.flatten(x, 1)
        x = self.fc(feat)
        return x, feat