import torch
from torch import nn
from models import register_model


class CCA(nn.Module):
    """
    CCA (Contrast-aware Channel Attention)
    Daje vaznost kanalima na osnovu kontrasta (varijanse)
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # mean i stdvar po (H, W): [B,C,H,W] -> [B,C,1,1]
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        std = torch.std(x, dim=(2, 3), keepdim=True, unbiased=False)
        # channel attention
        attn = self.excitation(mean + std)
        return x * attn


class IMDB(nn.Module):
    """
        IMDB (Information Multi-Distillation Block)

        4 destilacione konvolucije
        - Na izlazu svake konvolucije se features odvajaju na destilovane kanale i ostatak (fiksiran split)
        - Ostatak nastavlja dalje kroz sledeću konvoluciju.
        - Na kraju se spajaju svi destilovani delovi prolaze kroz attention (CCA) i 1x1 conv
        - Rezidualni izlaz
        """

    def __init__(self, in_ch, distillation_rate=0.25):
        super().__init__()

        self.act = nn.LeakyReLU(0.05)

        self.distilled_ch = int(in_ch * distillation_rate)
        self.remaining_ch = in_ch - self.distilled_ch

        # Prva konvolucija destilise sve ulazne kanale
        self.c1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1)
        # Ostale konvolucije destilisu ostatke
        self.c2 = nn.Conv2d(self.remaining_ch, in_ch, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(self.remaining_ch, in_ch, kernel_size=3, padding=1)
        # Zadnja konvolucija ne ostavlja ostatke
        self.c4 = nn.Conv2d(self.remaining_ch, self.distilled_ch, kernel_size=3, padding=1)
        # Attention
        self.cca = CCA(self.distilled_ch * 4)
        # 1x1 pointwise
        self.c5 = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)

    def forward(self, x):
        # Destilacija 1
        out_c1 = self.act(self.c1(x))
        distilled_c1 = out_c1[:, :self.distilled_ch, :, :]
        remaining_c1 = out_c1[:, self.distilled_ch:, :, :]
        # Destilacija 2
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2 = out_c2[:, :self.distilled_ch, :, :]
        remaining_c2 = out_c2[:, self.distilled_ch:, :, :]
        # Destilacija 3
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3 = out_c3[:, :self.distilled_ch, :, :]
        remaining_c3 = out_c3[:, self.distilled_ch:, :, :]
        # Destilacija 4
        distilled_c4 = self.act(self.c4(remaining_c3))
        # spajamo sve destilovane izlaze
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)
        # Attention + 1x1 pointwise + residual skip
        return self.c5(self.cca(out)) + x


class UpscaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upscale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


@register_model
class SR_IMDN(nn.Module):
    """
    IMDN (Information Multi-Distilation Network)
    Koristi destilaciju feature-a unutar svakog IMD bloka (IMDB)
    Izlazi svih blokova se na kraju konkateniraju i smanjuju 1x1 konvolucijom
    arhitektura:
    - 3x3 konvolucija da obradi RGB kanale i poveca broj kanala
    - IMDB blokovi
    - izlazi blokova se konkateniraju
    - 1x1 konvolucija smanjuje broj konkateniranih kanala
    - 3x3 konvolucija
    - PixelShuffle blok
    - za razliku od EDSR nema nista posle nego je pixel shuffle krajni sloj,
    """

    def __init__(self, num_blocks=6, nf=64, upscale_factor=2):
        super().__init__()
        # Prvo izvlacenje featura
        self.fea_conv = nn.Conv2d(3, nf, kernel_size=3, padding=1)
        # num_blocks IMDB blokova
        self.blocks = nn.ModuleList([IMDB(in_ch=nf) for _ in range(num_blocks)])
        # Obrada izlaza blokova
        self.conv = nn.Conv2d(nf * num_blocks, nf, kernel_size=1)
        self.lrelu = nn.LeakyReLU(0.05, inplace=True)
        # Rekonstrukcija u feature prostoru + skip
        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        # Pixel shuffle blok za kraj (izlaz 3 kanala)
        self.upscaler = UpscaleBlock(nf, 3, upscale_factor)

    def forward(self, x):
        out_fea = self.fea_conv(x)
        out = out_fea.clone()

        out_blocks = []
        for block in self.blocks:
            out = block(out)
            out_blocks.append(out)

        out_B = torch.cat(out_blocks, dim=1)
        out_B = self.lrelu(self.conv(out_B))

        out_lr = self.LR_conv(out_B) + out_fea

        return self.upscaler(out_lr)
