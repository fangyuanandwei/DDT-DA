import torch
import torch.nn as nn
import Model.BackBone as backbone

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

class Classifier(nn.Module):
    def __init__(self,number_classes=31):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(256, number_classes)

    def forward(self, input):
        fc_score = self.fc(input)
        return fc_score

    def get_parameters(self):
        parameters = [
             {'params':self.fc.parameters(),'lr_mult':10,'decay_mult':2}
        ]
        return parameters

