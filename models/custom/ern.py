import torch.nn as nn

from models import register_model


class ResidualExpansionBlock(nn.Module):
    """
    Rezidualni blok koji prvo obradjuje kanale sa 3x3 konvolucijom,
    in_channels --(3x3)-> in_channels
    Zatim radi ekspanziju kanala sa 1x1 konv
    in_channels --(1x1)-> 2*in_channels
    I na kraju vraca broj kanala sa 3x3 konv
    2*in_channels --(3x3)-> in_channels
    """

    def __init__(self, in_channels, expand_ratio=2):
        super().__init__()
        hidden_dim = in_channels * expand_ratio

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)


class UpscaleBlock(nn.Module):
    def __init__(self, in_ch, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


@register_model
class SR_CustomERN(nn.Module):
    """
    ERN - ExpansionResidualNetwork
    """

    def __init__(self, upscale_factor=2, num_blocks=4, width_mult=1.0):
        super().__init__()
        block_channels = int(64 * width_mult)
        last_channel = int(128 * width_mult)

        # prva 3x3 konvolucija da obradi RGB kanale
        self.stem = nn.Sequential(
            nn.Conv2d(3, block_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05, inplace=True)
        )

        # blokovi
        self.blocks = nn.ModuleList([ResidualExpansionBlock(in_channels=block_channels) for _ in range(num_blocks)])

        # 1x1 povecava broj kanala pre upscale dela
        self.conv_final = nn.Sequential(
            nn.Conv2d(block_channels, last_channel, kernel_size=1),
            nn.LeakyReLU(0.05, inplace=True)
        )

        self.upscaler = UpscaleBlock(in_ch=last_channel, upscale_factor=upscale_factor)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_final(x)
        x = self.upscaler(x)
        return x
