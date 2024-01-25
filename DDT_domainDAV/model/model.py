import torch
import torch.nn as nn
import model.BackBone as backbone

class Generator(nn.Module):
    def __init__(self, base_net='ResNet50'):
        super(Generator,self).__init__()
        self.sharedNet = backbone.network_dict[base_net]()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

    def forward(self, input):
        backbone_feature = self.sharedNet(input)

        #feature -> vector
        backbone_feature = self.avgpool(backbone_feature)
        backbone_feature = torch.flatten(backbone_feature, 1)

        feature_bottleneck = self.bottleneck(backbone_feature)
        return feature_bottleneck

    def get_parameters(self):
        parameters = [
             {'params':self.sharedNet.parameters(),'lr_mult':1,'decay_mult':1},
             {'params':self.bottleneck.parameters(),'lr_mult':10,'decay_mult':2}
        ]
        return parameters

    def out_features(self):
        out_features = 256
        return out_features