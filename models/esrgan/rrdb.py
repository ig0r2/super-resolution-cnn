import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model


class DenseLayer(nn.Module):
    """One conv layer inside a Dense Block (growth_channels output channels)."""

    def __init__(self, in_channels: int, growth_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class DenseBlock(nn.Module):
    """
    Residual Dense Block (RDB)
    Each layer concatenates all previous feature maps as input.
    Output is a residual: res_scale * F(x) + x
    """

    def __init__(self, nf: int = 64, gc: int = 32, res_scale: float = 0.2):
        # gc - growth_channels (medju kanali)
        super().__init__()
        self.res_scale = res_scale

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5 * self.res_scale + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block: 3 DenseBlocks with a residual skip.
    """

    def __init__(self, nf: int = 64, gc: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.rdb1 = DenseBlock(nf, gc, res_scale)
        self.rdb2 = DenseBlock(nf, gc, res_scale)
        self.rdb3 = DenseBlock(nf, gc, res_scale)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.res_scale + x


@register_model
class SR_ESRGAN(nn.Module):
    """
    ESRGAN RRDB (Residual-in-Residual Dense Block) Generator
    Paper: "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" Wang et al., 2018  (https://arxiv.org/abs/1809.00219)

    Args:
        upscale_factor:   2, 3 or 4 (paper implements only 4 as 2 + 2)
        nf:               base number of feature channels  (default 64)
        growth_channels:  growth channels inside each DenseBlock (default 32)
        num_blocks:       number of RRDB blocks (default 23)
    """

    def __init__(self, upscale_factor: int = 4, nf: int = 64, growth_channels: int = 32, num_blocks: int = 23):
        super().__init__()
        self.upscale_factor = upscale_factor
        # Shallow feature extraction
        self.conv_first = nn.Conv2d(3, nf, kernel_size=3, padding=1)

        # Deep feature extraction (RRDB trunk)
        self.trunk = nn.Sequential(*[RRDB(nf, growth_channels) for _ in range(num_blocks)])
        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)

        # Upsampling (via nn interpolation as in original implementation)
        if upscale_factor in (2, 3):
            self.upconv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        elif upscale_factor == 4:
            self.upconv = nn.Sequential(nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                                        nn.Conv2d(nf, nf, kernel_size=3, padding=1))

        # HR pass
        self.HRconv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.trunk(fea))
        fea = fea + trunk  # global residual

        fea = self.lrelu(self.upconv(F.interpolate(fea, scale_factor=self.upscale_factor, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
