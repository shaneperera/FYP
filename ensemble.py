import torch.nn as nn
from torchvision.models import resnet101
import torch
from densenet import densenet169


class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()

        self.densenet = densenet169(pretrained=True)
        self.densenet.load_state_dict(torch.load('models/best_model_1'))

        self.resnet = resnet101(pretrained=True)
        self.resnet.load_state_dict(torch.load(''))

    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.resnet(x)

        x = sum([x1, x2]) /2

        return x

