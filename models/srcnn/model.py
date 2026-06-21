from torch import nn
from models import register_model
import torch.nn.functional as F


@register_model
class SR_SRCNN(nn.Module):
    """
    SRCNN
    ali dodat padding
    i RGB
    """

    def __init__(self, upscale_factor):
        super().__init__()

        self.upscale_factor = upscale_factor
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
        )

    def forward(self, x, upscale_factor=None):
        if upscale_factor is None:
            upscale_factor = self.upscale_factor
        base = F.interpolate(x, scale_factor=upscale_factor, mode='bicubic', align_corners=False)
        return self.net(base)
