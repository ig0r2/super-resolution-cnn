from torch import nn
from models import register_model
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


@register_model
class SR_VDSR(nn.Module):
    """
    VDSR (Very Deep Super-Resolution)
    """

    def __init__(self, upscale_factor, num_blocks=18, nf=64):
        super().__init__()

        self.upscale_factor = upscale_factor
        self.net = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, padding=1),
            *[Block(nf) for _ in range(num_blocks)],
            nn.Conv2d(nf, 3, kernel_size=3, padding=1),
        )

    def forward(self, x, upscale_factor=None):
        if upscale_factor is None:
            upscale_factor = self.upscale_factor
        base = F.interpolate(x, scale_factor=upscale_factor, mode='bicubic', align_corners=False)
        return base + self.net(base)
