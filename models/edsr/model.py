from torch import nn
from models import register_model


class ResidualBlock(nn.Module):
    """
    Rezidualni blok sa 2 sloja 3x3 konvolucije, bez Batch normalizacije
    Takodje omogucava i koriscenje residual scaling za vece modele
    """

    def __init__(self, channels, res_scaling=1):
        super().__init__()
        self.res_scaling = res_scaling

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.res_scaling * self.layers(x)


class UpscaleBlock(nn.Module):
    """
    Konvolucija proizvodi [out channels]*(upscale^2) kanala
    PixelShuffle samo rasporedjuje piksele iz svih tih kanala u konacnu SR sliku

    primer za upscale_factor 2:
        umesto 1 piksela, u SR slici treba da se nadju 4 piksela (2x2 blok)
        Konvolucija proizvodi 4 puta vise kanala nego sto je izlaz
        PixelShuffle uzima trazena 4 piksela iz 4 razlicita kanala
    """

    def __init__(self, in_ch, out_ch, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


@register_model
class SR_EDSR(nn.Module):
    """
    EDSR (Enhanced Deep Super-Resolution)
    Koristi rezidualne blokove bez batch normalizaije
    arhitektura:
    - jedna 3x3 konvolucija da obradi RGB kanale i poveca broj kanala
    - rezidualni blokovi
    - jedna 3x3 konvolucija nakon blokova
    - PixelShuffle blok
    - jedna 3x3 konvolucija nakon upscalovane slike koja ce da pretvori izlaz u 3 kanala
    """

    def __init__(self, upscale_factor, num_blocks=16, nf=64, res_scaling=1):
        super().__init__()

        self.conv1 = nn.Conv2d(3, nf, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(nf, res_scaling) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.upscaler = UpscaleBlock(nf, nf, upscale_factor)
        self.conv3 = nn.Conv2d(nf, 3, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        residual = out

        for res_block in self.res_blocks:
            out = res_block(out)

        out = self.conv2(out)
        out += residual

        out = self.upscaler(out)
        out = self.conv3(out)

        return out


@register_model
class SR_EDSR_Multi(nn.Module):
    def __init__(self, num_blocks=16, nf=64, res_scaling=1, upscale_factor=2):
        super().__init__()

        self.upscale_factor = upscale_factor

        self.conv1 = nn.Conv2d(3, nf, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(nf, res_scaling) for _ in range(num_blocks)])
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1)

        self.upscalers = nn.ModuleDict({
            "2": UpscaleBlock(nf, nf, upscale_factor=2),
            "3": UpscaleBlock(nf, nf, upscale_factor=3),
            "4": UpscaleBlock(nf, nf, upscale_factor=4),
        })

        self.conv3 = nn.Conv2d(nf, 3, kernel_size=3, padding=1)

    def forward(self, x, upscale_factor=None):
        if upscale_factor is None:
            upscale_factor = self.upscale_factor

        out = self.conv1(x)
        residual = out

        for res_block in self.res_blocks:
            out = res_block(out)

        out = self.conv2(out)
        out += residual

        key = str(upscale_factor)
        if key not in self.upscalers:
            raise ValueError(f"Scale {upscale_factor} not supported.")

        out = self.upscalers[key](out)
        out = self.conv3(out)

        return out
