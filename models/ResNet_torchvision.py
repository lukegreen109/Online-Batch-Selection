# resnet: torchvision version

import torch
import torch.nn as nn
import torchvision.models as models

def create_model(m_type='resnet101',num_classes=1000, pretrained = False):
    # create various resnet models
    weights_map = {
        'resnet18': models.ResNet18_Weights.DEFAULT,
        'resnet50': models.ResNet50_Weights.DEFAULT,
        'resnet101': models.ResNet101_Weights.DEFAULT,
        'resnext50': models.ResNeXt50_32X4D_Weights.DEFAULT,
        'resnext101': models.ResNeXt101_32X8D_Weights.DEFAULT,
    }
    weights = weights_map[m_type] if pretrained else None

    if m_type == 'resnet18':
        model = models.resnet18(weights=weights)
    elif m_type == 'resnet50':
        model = models.resnet50(weights=weights)
    elif m_type == 'resnet101':
        model = models.resnet101(weights=weights)
    elif m_type == 'resnext50':
        model = models.resnext50_32x4d(weights=weights)
    elif m_type == 'resnext101':
        model = models.resnext101_32x8d(weights=weights)
    else:
        raise ValueError('Wrong Model Type')
        
    model = ResNet(model, num_classes)
    return model

class ResNet(nn.Module):
    def __init__(self, model, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = nn.Identity()
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)

    def forward(self, x, **kwargs):
        
        need_features = kwargs.get('need_features', False)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        
        x = self.layer1(x)        
        x = self.layer2(x)        
        x = self.layer3(x)        
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        x = self.fc(feat)
        if need_features:
            return x, feat
        else:
            return x

    def feat_nograd_forward(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)        
            x = self.layer1(x)        
            x = self.layer2(x)        
            x = self.layer3(x)        
            x = self.layer4(x)
            x = self.avgpool(x)
            feat = torch.flatten(x, 1)
        x = self.fc(feat)
        return x, feat