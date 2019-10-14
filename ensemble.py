import torch.nn as nn
from torchvision.models import resnet101
import torch
from densenet import densenet169
from resnet import wide_resnet101_2
import torch.utils.model_zoo as model_zoo


class Ensemble(nn.Module):
    def __init__(self, densenet_path,resnet_path):
        super(Ensemble, self).__init__()



        self.densenet = densenet169(pretrained=True, droprate= 0)
        self.densenet.load_state_dict(torch.load(densenet_path))

        self.resnet = wide_resnet101_2()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.resnet.load_state_dict(torch.load(resnet_path))


    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.resnet(x)

        x = sum([x1, x2]) /2

        return x

