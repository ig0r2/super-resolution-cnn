"""
ESRGAN loss functions.

Combines three terms:
  L_total = λ_pixel · L1  +  λ_perceptual · VGG  +  λ_adv · adversarial

VGG features are extracted from relu3_4 (before activation, as in the paper)
on images normalised to ImageNet statistics.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights


class VGGFeatureExtractor(nn.Module):
    """
    Extracts features from VGG19 relu3_4 (index 26 in vgg19.features).

    Inputs are expected in [0, 1]; they are then normalised to ImageNet stats
    """

    def __init__(self, feature_layer: int = 26):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer + 1])
        # Freeze
        for p in self.parameters():
            p.requires_grad = False
        # ImageNet statistics
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features((x - self.mean) / self.std)


class ESRGANLoss(nn.Module):
    """
    Combined generator loss for ESRGAN.

    Args:
        lambda_pixel:       weight for pixel-wise L1 loss
        lambda_perceptual:  weight for VGG perceptual loss
        lambda_adv:         weight for adversarial loss
        feature_layer:      which VGG layer to use (default 26 = relu3_4)
    """

    def __init__(self, lambda_pixel: float = 1e-2, lambda_perceptual: float = 1.0, lambda_adv: float = 5e-3,
                 feature_layer: int = 26):
        super().__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv

        self.vgg = VGGFeatureExtractor(feature_layer)
        self.l1 = nn.L1Loss()
        self.bce_logit = nn.BCEWithLogitsLoss()

    def pixel_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.l1(sr, hr)

    def perceptual_loss(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Clamp to [0,1] — VGG was trained on natural images
        sr_feat = self.vgg(sr.clamp(0, 1))
        hr_feat = self.vgg(hr.clamp(0, 1))
        return self.l1(sr_feat, hr_feat)

    # def generator_adv_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
    #     """SRGAN BCE loss that pushes D(G(x)) → 1."""
    #     real_labels = torch.ones_like(fake_logits)
    #     return self.bce_logit(fake_logits, real_labels)

    def generator_adv_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Relativistički loss za generator (RaGAN)."""
        # Generator želi da lažne slike budu iznad proseka pravih slika,
        # a prave slike ispod proseka lažnih.
        bce = nn.BCEWithLogitsLoss()
        loss_real = bce(real_logits - torch.mean(fake_logits), torch.zeros_like(real_logits))
        loss_fake = bce(fake_logits - torch.mean(real_logits), torch.ones_like(fake_logits))
        return (loss_real + loss_fake) * 0.5

    # ------------------------------------------------------------------
    # Combined generator loss (used as self.criterion in the trainer)
    # ------------------------------------------------------------------

    def generator_loss(self, sr: torch.Tensor, hr: torch.Tensor, real_logits: torch.Tensor,
                       fake_logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, dict_of_components) for logging.
        """
        l_pixel = self.pixel_loss(sr, hr)
        l_perc = self.perceptual_loss(sr, hr)
        l_adv = self.generator_adv_loss(real_logits, fake_logits)

        total = self.lambda_pixel * l_pixel + self.lambda_perceptual * l_perc + self.lambda_adv * l_adv

        return total, {'pixel': l_pixel.item(), 'perceptual': l_perc.item(), 'adversarial': l_adv.item(), }

    # @staticmethod
    # def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    #     """SRGAN discriminator loss"""
    #     bce = nn.BCEWithLogitsLoss()  # Binary Cross Entropy
    #     loss_real = bce(real_logits, torch.ones_like(real_logits))  # real -> 1
    #     loss_fake = bce(fake_logits, torch.zeros_like(fake_logits))  # fake -> 0
    #     return (loss_real + loss_fake) * 0.5

    @staticmethod
    def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Relativistic Average GAN (RaGAN) loss za ESRGAN diskriminator"""
        bce = nn.BCEWithLogitsLoss()
        # Relativna ocena za prave slike -> 1
        loss_real = bce(real_logits - torch.mean(fake_logits), torch.ones_like(real_logits))
        # Relativna ocena za lažne slike -> 0
        loss_fake = bce(fake_logits - torch.mean(real_logits), torch.zeros_like(fake_logits))
        return (loss_real + loss_fake) * 0.5
