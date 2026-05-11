import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_model(m_type='twolayerbinary', input_dim=[1,1,20], hidden_dim=15, output_dim=None, num_classes=None, pretrained=False, **kwargs):
    input_dim_scalar = math.prod(input_dim)
    if output_dim is None:
        output_dim = num_classes if num_classes is not None else 1
    if m_type == 'twolayerbinary':
        if output_dim != 1:
            raise ValueError('TwoLayerBinary requires output_dim=1.')
        model = TwoLayerBinary(input_dim=input_dim_scalar, hidden_dim=hidden_dim, output_dim=output_dim)
    else:
        model = TwoLayer(input_dim=input_dim_scalar, hidden_dim=hidden_dim, output_dim=output_dim)
    return model

class TwoLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayer, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,output_dim)

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
    
class TwoLayerBinary(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerBinary, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,output_dim)

    def forward(self, x, **kwargs):
        feature = kwargs.get('need_features', False)

        x = F.relu(self.linear1(x))
        feat = x
        x = self.linear2(x)
        zeros = torch.zeros_like(x)

        if feature:
            return torch.cat([zeros, x], dim=1), feat
        else:
            return torch.cat([zeros, x], dim=1)
    
    def feat_nograd_forward(self, x):
        with torch.no_grad():
            x = F.relu(self.linear1(x))
            feat = torch.flatten(x, 1)
        x = self.linear2(feat)
        zeros = torch.zeros_like(x)
        return torch.cat([zeros, x], dim=1), feat