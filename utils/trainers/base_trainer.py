"""
base_trainer.py — shared infrastructure for all trainer variants.

Responsibilities:
  - Build train/val DataLoaders (single-scale or multiscale, with optional JPEG degradation)
  - Own the epoch loop, scheduler step, and checkpoint save/load protocol
  - Expose set_model() so each run can swap in a fresh model without recreating loaders

Subclasses override:
  - _build_model_and_optim(config)  — create self.model, self.optimizer, self.scheduler, self.criterion
  - train_epoch()                   — one forward/backward pass over self.train_loader
  - validate()                      — one evaluation pass, returns a scalar score (higher = better)
  - _checkpoint_dict()              — extra keys to include when saving
  - _restore_checkpoint(ckpt)       — restore extra keys when loading
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import get_test_set
from datasets import get_training_set, TrainCollateFn, MultiscaleTrainCollateFn
from datasets.data import get_div2k_test_set_multi
from utils.metrics import SSIM
from utils.path import get_checkpoints_path
from utils.plot import plot_training_history


class BaseTrainer:
    """
    Owns DataLoaders, the epoch loop, and checkpoint I/O.
    Model-specific logic lives in subclasses.
    """

    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.multiscale = config['multiscale']
        jpeg_degradation = config['jpeg_degradation']

        patch_size = config['patch_size']
        num_workers = config['num_workers']
        batch_size = config['batch_size']

        if self.multiscale:
            assert patch_size % 12 == 0, "patch_size must be divisible by 2, 3, and 4 for multiscale training"
            self.train_loader = DataLoader(
                dataset=get_training_set(patch_size=patch_size, preload=config['train_preload']),
                collate_fn=MultiscaleTrainCollateFn(jpeg_degradation=jpeg_degradation),
                num_workers=num_workers, batch_size=batch_size,
                shuffle=True, persistent_workers=True, pin_memory=True,
            )
            self.val_loader = DataLoader(
                dataset=get_div2k_test_set_multi(
                    preload=config['val_preload'], normalize=True, jpeg_degradation=jpeg_degradation),
                num_workers=1, batch_size=1, persistent_workers=True, pin_memory=True,
            )
            self._upscale_factor = None  # determined per-batch during multiscale val
        else:
            self._upscale_factor = config['model']['params'].get('upscale_factor', 1)
            assert patch_size % self._upscale_factor == 0, \
                f"patch_size must be divisible by upscale_factor ({self._upscale_factor})"
            self.train_loader = DataLoader(
                dataset=get_training_set(patch_size=patch_size, preload=config['train_preload']),
                collate_fn=TrainCollateFn(self._upscale_factor, jpeg_degradation=jpeg_degradation),
                num_workers=num_workers, batch_size=batch_size,
                shuffle=True, persistent_workers=True, pin_memory=True,
            )
            self.val_loader = DataLoader(
                dataset=get_test_set(
                    name="DIV2K", upscale_factor=self._upscale_factor,
                    preload=config['val_preload'], jpeg_degradation=jpeg_degradation),
                num_workers=1, batch_size=1, persistent_workers=True, pin_memory=True,
            )

        # These are set by set_model() before each training run.
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.config = None
        self.epoch = 1
        self.epochs = config['epochs']
        self.best_val_score = 0
        self.history = None
        self.metrics_ssim = SSIM(device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_model(self, config: dict) -> "BaseTrainer":
        """
        (Re-)initialise model, optimiser, scheduler, and history for one training run.
        Call once per entry in config['models'].
        """
        self.config = config
        self.epoch = 1
        self.best_val_score = 0
        self.metrics_ssim.reset()
        self.history = {'training': [], 'validation': []}
        self._build_model_and_optim(config)
        return self

    def train(self):
        print(f"Training: {self.config['model']['checkpoint_name']}")
        self._load_checkpoint()

        while self.epoch <= self.epochs:
            self.train_epoch()
            self.scheduler.step()

            if self.epoch % self.config['validate_every'] == 0 or self.epoch == self.epochs:
                score = self.validate()
                if score > self.best_val_score:
                    self.best_val_score = score
                    self._save_checkpoint()  # best
                self._save_checkpoint(suffix='_latest')

            self.epoch += 1

        plot_training_history(self.history, self.config['model']['checkpoint_name'])

    # ------------------------------------------------------------------
    # Subclass interface — override these
    # ------------------------------------------------------------------

    def _build_model_and_optim(self, config: dict):
        """Create self.model, self.optimizer, self.scheduler, self.criterion."""
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def validate(self) -> float:
        """Run validation; return a scalar score (higher = better)."""
        raise NotImplementedError

    def _checkpoint_dict(self) -> dict:
        """Extra keys to persist in the checkpoint (e.g. discriminator state)."""
        return {}

    def _restore_checkpoint(self, ckpt: dict):
        """Restore extra keys from a loaded checkpoint."""

    # ------------------------------------------------------------------
    # Checkpoint helpers (shared)
    # ------------------------------------------------------------------

    def _ckpt_path(self, suffix: str = "") -> Path:
        name = self.config['model']['checkpoint_name']
        prefix = "multiscale" if self.multiscale else f"{self._upscale_factor}x"
        return get_checkpoints_path(f"{prefix}/{name}{suffix}.pth")

    def _save_checkpoint(self, suffix: str = ""):
        path = self._ckpt_path(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': self.epoch,
            'best_val_score': self.best_val_score,
            'history': self.history,
            'model_config': self.config['model'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            **self._checkpoint_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def _load_checkpoint(self):
        path = self._ckpt_path('_latest')
        if not path.exists():
            print("No checkpoint found")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epoch = ckpt['epoch'] + 1
        self.best_val_score = ckpt['best_val_score']
        self.history = ckpt['history']
        self._restore_checkpoint(ckpt)
        print(f"Resumed from epoch {ckpt['epoch']}")
