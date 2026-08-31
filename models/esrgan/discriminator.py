"""
VGG-style Discriminator for ESRGAN.

Classifies 128×128 (or larger) HR patches as real / fake.
Architecture follows the original paper exactly.
"""

import torch.nn as nn


def _conv_block(in_ch: int, out_ch: int, stride: int = 1, bn: bool = True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class VGGDiscriminator(nn.Module):
    """
    VGG-style discriminator.

    Input:  (B, 3, H, W)  — full-resolution HR patches (e.g. 128×128)
    Output: (B, 1)        — real/fake logit per image

    The 8 conv-blocks progressively halve spatial resolution while doubling
    channels, matching the original ESRGAN paper.
    """

    def __init__(self, input_size: int = 128, nf: int = 64):
        super().__init__()

        # -------------------------------------------------------------------
        # Feature extractor — 8 conv blocks
        # stride=1 keeps size, stride=2 halves size
        # -------------------------------------------------------------------
        self.features = nn.Sequential(
            # Block 1  — no BN on first layer (common practice)
            *_conv_block(3, nf, stride=1, bn=False),  # 128
            *_conv_block(nf, nf, stride=2, bn=True),  # 64
            # Block 2
            *_conv_block(nf, nf * 2, stride=1, bn=True),  # 64
            *_conv_block(nf * 2, nf * 2, stride=2, bn=True),  # 32
            # Block 3
            *_conv_block(nf * 2, nf * 4, stride=1, bn=True),  # 32
            *_conv_block(nf * 4, nf * 4, stride=2, bn=True),  # 16
            # Block 4
            *_conv_block(nf * 4, nf * 8, stride=1, bn=True),  # 16
            *_conv_block(nf * 8, nf * 8, stride=2, bn=True),  # 8
        )

        # -------------------------------------------------------------------
        # Classifier head
        # After 4× stride-2 blocks the spatial size is input_size // 16.
        # For input_size=128 → 8×8 → 512 channels → flat 512*8*8 = 32 768
        # -------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(input_size // 16),  # safety for non-128 inputs
            nn.Flatten(),
            nn.Linear(nf * 8 * (input_size // 16) ** 2, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))
