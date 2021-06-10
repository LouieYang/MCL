import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.registry as registry

@registry.Encoder.register("FourLayer_64F")
class FourLayer_64F(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(FourLayer_64F, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1,bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(0.2, True))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0,bias=False))
        self.out_channels = 64

        for l in self.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.xavier_uniform_(l.weight)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)
            elif isinstance(l, nn.Linear):
                torch.nn.init.normal_(l.weight, 0, 0.01)
                if l.bias is not None:
                    torch.nn.init.constant(l.bias, 0)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64
