"""
trainer.py — L1-loss PSNR trainer (fixed-scale and multiscale).

Multiscale behaviour is activated via the `multiscale` flag passed to __init__.
The DataLoaders are built by BaseTrainer; this class only adds the
model/optimiser setup and the train/validate logic.
"""

import sys

import torch
from torch import nn, optim
from tqdm import tqdm

from utils.trainers.base_trainer import BaseTrainer
from models import get_model
from utils.model_utils import tile_forward


class Trainer(BaseTrainer):

    # ------------------------------------------------------------------
    # Model / optimiser setup
    # ------------------------------------------------------------------

    def _build_model_and_optim(self, config: dict):
        self.model = get_model(config['model']).to(self.device)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=config['scheduler_milestones'], gamma=0.5)

    def train_epoch(self):
        epoch_loss = 0.0
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f"Epoch [{self.epoch}/{self.epochs}]", file=sys.stdout)

        scale = None
        for i, batch in enumerate(pbar, 1):
            if self.multiscale:
                lr, hr, scale = batch
            else:
                lr, hr = batch

            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            output = self.model(lr, scale) if self.multiscale else self.model(lr)
            loss = self.criterion(output, hr)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{epoch_loss / i:.4f}"})

        self.history["training"].append({
            "epoch": self.epoch,
            "loss": epoch_loss / len(self.train_loader),
        })

    def validate(self) -> float:
        val_loss = 0.0
        self.metrics_ssim.reset()
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
                        val_loss += self.criterion(out, hr).item()
                        self.metrics_ssim.update(out, hr)

                    pbar.set_postfix({'SSIM': f'{self.metrics_ssim.compute():.4f}'})
                val_loss /= len(self.val_loader) * 3
            else:
                for i, (lr, hr) in enumerate(pbar, 1):
                    lr = lr.to(self.device, non_blocking=True)
                    hr = hr.to(self.device, non_blocking=True)

                    out = tile_forward(self.model, self._upscale_factor, lr, tile_size=256, overlap=8)
                    val_loss += self.criterion(out, hr).item()
                    self.metrics_ssim.update(out, hr)

                    pbar.set_postfix({'SSIM': f'{self.metrics_ssim.compute():.4f}'})
                val_loss /= len(self.val_loader)

        ssim = self.metrics_ssim.compute().item()
        self.history['validation'].append({
            'epoch': self.epoch,
            'loss': val_loss,
            'ssim': ssim,
        })
        return ssim
