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


class Block7(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=7, padding=3),
        )

    def forward(self, x):
        return x + self.layers(x)


class Block_prelu(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels, 0),
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


@register_model
class SR_FastEDSR_Multi(nn.Module):
    def __init__(self, num_blocks=1, nf=4, upscale_factor=2):
        super().__init__()

        self.upscale_factor = upscale_factor

        self.net = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, padding=1),
            *[Block(nf) for _ in range(num_blocks)],
        )

        self.upscale_block_2 = nn.Sequential(
            nn.Conv2d(nf, 3 * (2 ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        self.upscale_block_3 = nn.Sequential(
            nn.Conv2d(nf, 3 * (3 ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(3)
        )
        self.upscale_block_4 = nn.Sequential(
            nn.Conv2d(nf, 3 * (4 ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(4)
        )

    def forward(self, x, upscale_factor=None):
        if upscale_factor is None:
            upscale_factor = self.upscale_factor

        base = F.interpolate(x, scale_factor=upscale_factor, mode='bilinear', align_corners=False)

        fea = self.net(x)

        if upscale_factor == 2:
            res = self.upscale_block_2(fea)
        elif upscale_factor == 3:
            res = self.upscale_block_3(fea)
        elif upscale_factor == 4:
            res = self.upscale_block_4(fea)
        else:
            raise ValueError(f"Scale {upscale_factor} not supported.")

        return base + res


@register_model
class SR_FastEDSR_1x(nn.Module):
    def __init__(self, num_blocks=1, nf=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, padding=1),
            *[Block(nf) for _ in range(num_blocks)],
            nn.Conv2d(nf, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)
