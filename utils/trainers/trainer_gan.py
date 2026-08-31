"""
trainer_gan.py — ESRGAN GAN fine-tuning trainer.

Extends BaseTrainer with:
  - A VGGDiscriminator alongside the generator
  - ESRGANLoss (pixel + perceptual + adversarial)
  - LPIPS validation metric
  - Combined score = (1-LPIPS)*α + SSIM*(1-α)  used for checkpoint selection
  - Full multiscale and JPEG degradation support (inherited from BaseTrainer)

Two-phase training workflow
──────────────────────────
  Phase 1  PSNR pre-training:  use Trainer with the same config.
  Phase 2  GAN fine-tuning:    create TrainerGAN, call set_model(config) where
           config['model'] either names a fresh model or provides
           'pretrain_checkpoint_path' to warm-start from Phase 1 weights.
"""

import sys

import torch
from torch import optim
from tqdm import tqdm

from utils.plot import plot_training_history_gan
from utils.trainers.base_trainer import BaseTrainer
from loss.esrgan_loss import ESRGANLoss
from models.esrgan.discriminator import VGGDiscriminator
from utils.checkpoints import load_model_from_checkpoint
from utils.metrics import LPIPS
from utils.model_utils import tile_forward
from utils.path import get_checkpoints_path


class TrainerGAN(BaseTrainer):

    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)

        self.discriminator = None
        self.d_optimizer = None
        self.d_scheduler = None

        self.metrics_lpips = LPIPS(device=device)
        self.score_alpha = 0.8  # weight of (1-LPIPS) in composite validation score
        self.best_val_score = 0.0

    # ------------------------------------------------------------------
    # Model / optimiser setup
    # ------------------------------------------------------------------

    def _build_model_and_optim(self, config: dict):
        # Generator: resume from a named checkpoint or warm-start from PSNR weights.
        ckpt_path = self._ckpt_path('_latest')
        if ckpt_path.exists():
            self.model, self.config['model'] = load_model_from_checkpoint(ckpt_path, self.device)
        else:
            gan_checkpoint_name = config['model']['checkpoint_name']
            self.model, self.config['model'] = load_model_from_checkpoint(
                get_checkpoints_path(config['model']['pretrain_checkpoint_path']), self.device)
            self.config['model']['checkpoint_name'] = gan_checkpoint_name

        self.criterion = ESRGANLoss(lambda_pixel=1e-2, lambda_perceptual=1.0, lambda_adv=5e-3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config['scheduler_milestones'], gamma=0.5)

        # Discriminator
        self.discriminator = VGGDiscriminator(input_size=int(config['patch_size']), nf=64).to(self.device)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=config['lr'])
        self.d_scheduler = optim.lr_scheduler.MultiStepLR(
            self.d_optimizer, milestones=config['scheduler_milestones'], gamma=0.5)

    def train_epoch(self):
        g_loss_sum = d_loss_sum = 0.0
        pixel_sum = perc_sum = adv_sum = 0.0

        self.model.train()
        self.discriminator.train()
        pbar = tqdm(self.train_loader, desc=f"Epoch [{self.epoch}/{self.epochs}]", file=sys.stdout)

        scale = None
        for i, batch in enumerate(pbar, 1):
            if self.multiscale:
                lr, hr, scale = batch
            else:
                lr, hr = batch

            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)

            sr = self.model(lr, scale) if self.multiscale else self.model(lr)

            real_logits = self.discriminator(hr)
            fake_logits = self.discriminator(sr)

            self.d_optimizer.zero_grad()
            self.optimizer.zero_grad()

            d_loss = self.criterion.discriminator_loss(real_logits, fake_logits.detach())
            g_loss, components = self.criterion.generator_loss(sr, hr, real_logits.detach(), fake_logits)

            d_loss.backward()
            g_loss.backward()
            self.d_optimizer.step()
            self.optimizer.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            pixel_sum += components['pixel']
            perc_sum += components['perceptual']
            adv_sum += components['adversarial']
            pbar.set_postfix({'G': f'{g_loss_sum / i:.4f}', 'D': f'{d_loss_sum / i:.4f}'})

        n = len(self.train_loader)
        self.history['training'].append({
            'epoch': self.epoch,
            'g_loss': g_loss_sum / n,
            'd_loss': d_loss_sum / n,
            'pixel': pixel_sum / n,
            'perceptual': perc_sum / n,
            'adversarial': adv_sum / n,
        })

    def validate(self) -> float:
        self.metrics_ssim.reset()
        self.metrics_lpips.reset()
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", file=sys.stdout)

            if self.multiscale:
                for i, (lr_2, lr_3, lr_4, hr) in enumerate(pbar, 1):
                    hr = hr.to(self.device, non_blocking=True)
                    for scale, lr in ((2, lr_2), (3, lr_3), (4, lr_4)):
                        lr = lr.to(self.device, non_blocking=True)
                        self.model.upscale_factor = scale
                        out = tile_forward(self.model, scale, lr, tile_size=256, overlap=8)
                        self.metrics_ssim.update(out, hr)
                        self.metrics_lpips.update(out.clip(0, 1), hr)
                    pbar.set_postfix({'LPIPS': f'{self.metrics_lpips.compute():.4f}'})
            else:
                for i, (lr, hr) in enumerate(pbar, 1):
                    lr = lr.to(self.device, non_blocking=True)
                    hr = hr.to(self.device, non_blocking=True)
                    out = tile_forward(self.model, self._upscale_factor, lr, tile_size=256, overlap=8)
                    self.metrics_ssim.update(out, hr)
                    self.metrics_lpips.update(out.clip(0, 1), hr)
                    pbar.set_postfix({'LPIPS': f'{self.metrics_lpips.compute():.4f}'})

        ssim = self.metrics_ssim.compute().item()
        lpips = self.metrics_lpips.compute().item()
        score = (1 - lpips) * self.score_alpha + ssim * (1 - self.score_alpha)

        self.history['validation'].append({
            'epoch': self.epoch,
            'ssim': ssim,
            'lpips': lpips,
            'score': score,
        })
        return score

    # ------------------------------------------------------------------
    # Checkpoint extras (discriminator state)
    # ------------------------------------------------------------------

    def _checkpoint_dict(self) -> dict:
        return {
            'd_state_dict': self.discriminator.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
        }

    def _restore_checkpoint(self, ckpt: dict):
        if 'd_state_dict' in ckpt:
            self.discriminator.load_state_dict(ckpt['d_state_dict'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer_state_dict'])
            self.d_scheduler.load_state_dict(ckpt['d_scheduler_state_dict'])

    # ------------------------------------------------------------------
    # GAN needs two scheduler steps per epoch
    # ------------------------------------------------------------------

    def train(self):
        print(f"GAN training: {self.config['model']['checkpoint_name']}")
        self._load_checkpoint()

        while self.epoch <= self.epochs:
            self.train_epoch()
            self.scheduler.step()
            self.d_scheduler.step()

            if self.epoch % self.config['validate_every'] == 0 or self.epoch == self.epochs:
                score = self.validate()
                if score > self.best_val_score:
                    self.best_val_score = score
                    self._save_checkpoint()
                self._save_checkpoint(suffix='_latest')

            self.epoch += 1

        plot_training_history_gan(self.history, self.config['model']['checkpoint_name'])
