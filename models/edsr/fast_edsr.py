import torch.nn as nn
import torch.nn.functional as F

from models import register_model


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


@register_model
class SR_FastEDSR(nn.Module):
    """
    Uklonjena konvolucija nakon PixelShuffle kako se ne bi radila nad velikom slikom.
    Samim tim konvolucija pre PixelShuffle treba sada samo 3*r^2 kanala da ima na izlazu.
    Uklonjena residualna konekcija iz prvog sloja.
    Promenjeno da model predvidja rezidual nad vec upscaleovanom slikom pomocu bilinear metoda,
    cime dobijamo da manji modeli mogu da daju vrlo dobre rezultate.
    """

    def __init__(self, upscale_factor=2, num_blocks=1, nf=4):
        super().__init__()

        self.upscale_factor = upscale_factor

        self.net = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, padding=1),
            *[Block(nf) for _ in range(num_blocks)],
            nn.Conv2d(nf, 3 * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        return base + self.net(x)
