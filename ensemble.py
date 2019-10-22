import torch.nn as nn
from torchvision.models import resnet101
import torch
from densenet import densenet169
from resnet import resnet101,resnet152
import torch.utils.model_zoo as model_zoo
from VGG import vgg19_bn
from shufflenet import shufflenet_v2_x1_0
import numpy as np


class Ensemble(nn.Module):
    def __init__(self, densenet_path,resnet_path,vgg_path):
        super(Ensemble, self).__init__()

        self.densenet = densenet169(pretrained=True, droprate= 0)
        self.densenet.load_state_dict(torch.load(densenet_path))

        self.resnet = resnet101()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.resnet.load_state_dict(torch.load(resnet_path))

        self.vgg = vgg19_bn()
        self.vgg.classifier[6] = nn.Linear(4096, 1)
        self.vgg.load_state_dict(torch.load(vgg_path))

        # self.shufflenet = shufflenet_v2_x1_0()
        # num_ftrs = self.shufflenet.fc.in_features
        # self.shufflenet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.resnet(x)
        x3 = self.vgg(x)

        # x = sum([x1, x2, x3]) /3

        return x1,x2,x3
        # return x

class Ensemble2(nn.Module):
    def __init__(self):
        super(Ensemble2, self).__init__()

        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)


    def forward(self, x1,x2,x3):
        x = self.fc1(torch.tensor([x1,x2,x3]).cuda())
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x

class Ensemble3(nn.Module):
    def __init__(self):
        super(Ensemble3, self).__init__()

        self.conv1 = nn.Conv3d(4224, 1, 3)
        self.conv2 = nn.Conv2d(3, 1, 1) # input channels always changing here
        self.fc1 = nn.Linear(25, 1)


    def forward(self, x1,x2,x3):
        x = torch.cat((x1,x2,x3),dim=1)
        # x = torch.unsqueeze(x,0)
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = x.permute(1,0,2,3)
        x = self.conv2(x)
        x = torch.flatten(x)
        print(x.size())
        x = self.fc1(x)
        x = torch.sigmoid(x)

        return x