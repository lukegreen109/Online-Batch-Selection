import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model(m_type='twolayer', in_channels = 1, num_classes=10, pretrained=False):
    model = TwoLayer(input_dim=in_channels, output_dim=num_classes)  # default model
    return model

class TwoLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TwoLayer, self).__init__()

        self.linear1 = nn.Linear(input_dim, 28)
        self.linear2 = nn.Linear(28,output_dim)

    def forward(self, x, **kwargs):
        feature = kwargs.get('need_features', False)

        x = F.relu(self.linear1(x))
        feat = x
        x = self.linear2(x)

        if feature:
            return x, feat
        else:
            return x
    
    def feat_nograd_forward(self, x):
        with torch.no_grad():
            x = F.relu(self.linear1(x))
            feat = torch.flatten(x, 1)
        x = self.linear2(feat)
        return x, feat