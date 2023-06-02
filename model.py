import torchvision
import utils
from torch import nn


def getnet(layers_num, device):
    pretrained_net = torchvision.models.vgg19(pretrained=True)
    net = nn.Sequential(*[pretrained_net.features[i]
                          for i in range(layers_num)]).to(device)
    return net
