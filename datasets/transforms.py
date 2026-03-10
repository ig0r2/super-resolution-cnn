import random

from torch import nn
from torchvision.transforms import v2


class RandomRotation90(nn.Module):
    def __init__(self):
        super(RandomRotation90, self).__init__()

    def forward(self, img):
        angle = 90
        if random.random() < 0.25:
            return img
        elif random.random() < 0.5:
            angle = 180
        elif random.random() < 0.75:
            angle = 270
        return v2.functional.rotate(img, angle)
