import torch
from torch import nn
import torch.nn.functional as F

from models import register_model


class ESA(nn.Module):
    """
    ESA (Enhanced Spatial Attention)
    spatial attention - uci prostorno koji pikseli su vazni
    Cilj - Naučiti masku koja naglašava važne regione slike (ivice, teksture, detalje)
    smanji rezoluciju -> max pool -> obradi -> bilinear upsale back -> residual -> mask
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = channels // reduction

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(channels, reduced, kernel_size=1)  # channel reduction
        self.conv2 = nn.Conv2d(reduced, reduced, kernel_size=3, stride=2)  # Downsampling to increase receptive field
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = nn.Conv2d(reduced, reduced, kernel_size=3, padding=1)  # Local refinement after pooling
        self.conv3 = nn.Conv2d(reduced, reduced, kernel_size=3, padding=1)  # nonlinear refinement
        self.conv3_ = nn.Conv2d(reduced, reduced, kernel_size=3, padding=1)  # nonlinear refinement

        self.conv_f = nn.Conv2d(reduced, reduced, kernel_size=1)

        self.conv4 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = self.conv1(x)

        v_max = self.max_pool(self.conv2(c1_))
        v_range = self.relu(self.conv_max(v_max))

        c3 = self.conv3_(self.relu(self.conv3(v_range)))
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        cf = self.conv_f(c1_)  # skip connection
        m = self.sigmoid(self.conv4(c3 + cf))

        return x * m


class RFDB(nn.Module):
    """
    ulaz u jedan nivo destilacije prolazi paralelno kroz 1x1 i SRB
    - 1x1 konvolucija prvo odredjuje destilovane kanale iz ulaza
    - Ostatak se iz ulaza bira koristeci SRB (Shallow Residual Block, 3x3 konv + skip)
    """

    def __init__(self, in_ch):
        super().__init__()

        self.act = nn.LeakyReLU(0.05)

        self.distilled_ch = in_ch // 2
        self.remaining_ch = in_ch

        self.c1_d = nn.Conv2d(in_ch, self.distilled_ch, kernel_size=1)
        self.c1_r = nn.Conv2d(in_ch, self.remaining_ch, kernel_size=3, padding=1)

        self.c2_d = nn.Conv2d(self.remaining_ch, self.distilled_ch, kernel_size=1)
        self.c2_r = nn.Conv2d(self.remaining_ch, self.remaining_ch, kernel_size=3, padding=1)

        self.c3_d = nn.Conv2d(self.remaining_ch, self.distilled_ch, kernel_size=1)
        self.c3_r = nn.Conv2d(self.remaining_ch, self.remaining_ch, kernel_size=3, padding=1)

        self.c4 = nn.Conv2d(self.remaining_ch, self.distilled_ch, kernel_size=3, padding=1)

        self.c_out = nn.Conv2d(self.distilled_ch * 4, in_ch, kernel_size=1)

        self.esa = ESA(in_ch)

    def forward(self, x):
        # Destilacija 1
        distilled_c1 = self.act(self.c1_d(x))
        remaining_c1 = self.act(self.c1_r(x) + x)
        # Destilacija 2
        distilled_c2 = self.act(self.c2_d(remaining_c1))
        remaining_c2 = self.act(self.c2_r(remaining_c1) + remaining_c1)
        # Destilacija 3
        distilled_c3 = self.act(self.c3_d(remaining_c2))
        remaining_c3 = self.act(self.c3_r(remaining_c2) + remaining_c2)
        # Destilacija 4
        distilled_c4 = self.act(self.c4(remaining_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)
        return self.esa(self.c_out(out))


class UpscaleBlock(nn.Module):
    def __init__(self, in_ch, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


@register_model
class SR_RFDN(nn.Module):
    """
    (pogledati prvo IMDN)
    RFDN (Residual Feature Destilation Network)
    RFDN je direktno unapredjenje IMDN mreze.
    RFDN dozvoljava mrezi da nauci sama koje feature da destiluje.
    RFDN koristi ESA (Enhanced Spatial Attention) koji uci prostorni attention medju piskelima.
    """

    def __init__(self, num_blocks=6, nf=48, upscale_factor=2):
        # spoljna arhitektura ista kao kod IMDN, samo se blokovi razlikuju
        super().__init__()

        self.fea_conv = nn.Conv2d(3, nf, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([RFDB(in_ch=nf) for _ in range(num_blocks)])

        self.conv = nn.Conv2d(nf * num_blocks, nf, kernel_size=1)
        self.lrelu = nn.LeakyReLU(0.05, inplace=True)

        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)

        self.upscaler = UpscaleBlock(in_ch=nf, upscale_factor=upscale_factor)

    def forward(self, x):
        out_fea = self.fea_conv(x)

        out = out_fea
        out_blocks = []
        for block in self.blocks:
            out = block(out)
            out_blocks.append(out)

        out_B = torch.cat(out_blocks, dim=1)
        out_B = self.lrelu(self.conv(out_B))

        out_lr = self.LR_conv(out_B) + out_fea

        return self.upscaler(out_lr)


@register_model
class SR_RFDN_Multi(nn.Module):
    def __init__(self, num_blocks=6, nf=48, upscale_factor=2):
        super().__init__()

        self.upscale_factor = upscale_factor

        self.fea_conv = nn.Conv2d(3, nf, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([RFDB(in_ch=nf) for _ in range(num_blocks)])

        self.conv = nn.Conv2d(nf * num_blocks, nf, kernel_size=1)
        self.lrelu = nn.LeakyReLU(0.05, inplace=True)

        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)

        self.upscalers = nn.ModuleDict({
            "2": UpscaleBlock(in_ch=nf, upscale_factor=2),
            "3": UpscaleBlock(in_ch=nf, upscale_factor=3),
            "4": UpscaleBlock(in_ch=nf, upscale_factor=4),
        })

    def forward(self, x, upscale_factor=None):
        if upscale_factor is None:
            upscale_factor = self.upscale_factor

        out_fea = self.fea_conv(x)

        out = out_fea
        out_blocks = []
        for block in self.blocks:
            out = block(out)
            out_blocks.append(out)

        out_B = self.lrelu(self.conv(torch.cat(out_blocks, dim=1)))
        out_lr = self.LR_conv(out_B) + out_fea

        key = str(upscale_factor)
        if key not in self.upscalers:
            raise ValueError(f"Scale {upscale_factor} not supported.")

        return self.upscalers[key](out_lr)
