from torch import nn
from models import register_model


class AnchorUpscaleBlock(nn.Module):
    """
    Rep ABPN-a: dve konvolucije do 3*r^2 kanala, dodaje se anchor (ulaz ponovljen r^2 puta)
    pa PixelShuffle. Anchor posle PixelShuffle daje nearest-neighbor upscale ulaza,
    tako da mreza uci samo rezidual nad njim.
    """

    def __init__(self, in_ch, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor
        out_ch = 3 * (upscale_factor ** 2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, fea, x):
        # PixelShuffle ocekuje kanale kao [C, r, r], pa se svaki RGB kanal ponavlja r^2 puta uzastopno
        anchor = x.repeat_interleave(self.upscale_factor ** 2, dim=1)
        out = self.conv2(self.act(self.conv1(fea)))
        return self.pixel_shuffle(out + anchor)


def _body(nf, num_blocks):
    layers = [nn.Conv2d(3, nf, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    for _ in range(num_blocks):
        layers += [nn.Conv2d(nf, nf, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


@register_model
class SR_ABPN(nn.Module):
    """
    ABPN (Anchor-Based Plain Net, Du et al. 2021, MAI 2021 challenge)
    Plain mreza bez skip konekcija u telu, pogodna za mobilne/INT8 akceleratore.
    arhitektura:
    - 3x3 konvolucija 3 -> nf, ReLU
    - num_blocks x (3x3 konvolucija nf -> nf, ReLU)
    - 3x3 konvolucija nf -> 3*r^2, ReLU
    - 3x3 konvolucija 3*r^2 -> 3*r^2
    - + anchor (ulaz ponovljen r^2 puta) i PixelShuffle
    Original: nf=28, num_blocks=4 i clip na izlazu (ovde bez clip-a, kao ostali modeli).
    """

    def __init__(self, upscale_factor=2, num_blocks=4, nf=28):
        super().__init__()

        self.upscale_factor = upscale_factor

        self.body = _body(nf, num_blocks)
        self.upscaler = AnchorUpscaleBlock(nf, upscale_factor)

    def forward(self, x):
        return self.upscaler(self.body(x), x)


@register_model
class SR_ABPN_Multi(nn.Module):
    """
    ABPN sa zajednickim telom i posebnim anchor repom za svaki scale (2, 3, 4).
    """

    def __init__(self, num_blocks=4, nf=28, upscale_factor=2):
        super().__init__()

        self.upscale_factor = upscale_factor

        self.body = _body(nf, num_blocks)

        self.upscalers = nn.ModuleDict({
            "2": AnchorUpscaleBlock(in_ch=nf, upscale_factor=2),
            "3": AnchorUpscaleBlock(in_ch=nf, upscale_factor=3),
            "4": AnchorUpscaleBlock(in_ch=nf, upscale_factor=4),
        })

    def forward(self, x, upscale_factor=None):
        if upscale_factor is None:
            upscale_factor = self.upscale_factor

        key = str(upscale_factor)
        if key not in self.upscalers:
            raise ValueError(f"Scale {upscale_factor} not supported.")

        return self.upscalers[key](self.body(x), x)
