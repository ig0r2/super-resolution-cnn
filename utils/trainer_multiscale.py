import sys
from os.path import exists

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from datasets import get_training_set, MultiscaleTrainCollateFn
from datasets.data import get_div2k_test_set_multi
from utils.metrics import SSIM
from utils.model_utils import tile_forward
from utils.plot import plot_training_history


class TrainerMultiscale:
    def __init__(self, config, device, jpeg_degradation=False):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = device
        self.config = config

        assert config['patch_size'] % 12 == 0, "Patch size not divisible by 2,3,4"

        self.train_loader = DataLoader(
            dataset=get_training_set(patch_size=config['patch_size'], preload=config['train_preload']),
            collate_fn=MultiscaleTrainCollateFn(jpeg_degradation=jpeg_degradation),
            num_workers=config['num_workers'], batch_size=config['batch_size'],
            shuffle=True, persistent_workers=True, pin_memory=True
        )
        self.val_loader = DataLoader(
            dataset=get_div2k_test_set_multi(preload=config['val_preload'], normalize=True,
                                             jpeg_degradation=jpeg_degradation),
            num_workers=1, batch_size=1, persistent_workers=True, pin_memory=True
        )

        self.epoch = 1
        self.epochs = config['epochs']

        self.best_val_ssim = 0
        self.metrics_ssim = SSIM(device=device)

        self.history = None

    def set_model(self, model, optimizer, scheduler, criterion, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.epoch = 1
        self.best_val_ssim = 0
        self.metrics_ssim.reset()
        self.history = {'training': [], 'validation': []}

        return self

    def train_epoch(self):
        epoch_loss = 0
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f'Epoch [{self.epoch}/{self.epochs}]', file=sys.stdout)
        for iteration, (input, target, scale) in enumerate(pbar, 1):
            input, target = input.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(input, scale), target)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            pbar.set_postfix({'Loss': f'{epoch_loss / iteration:.4f}'})

        self.history['training'].append({
            'epoch': self.epoch,
            'loss': epoch_loss / len(self.train_loader)
        })

    def validate(self):
        val_loss = 0
        self.metrics_ssim.reset()
        self.model.eval()

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validation', file=sys.stdout)
            for iteration, (lr_2, lr_3, lr_4, target) in enumerate(pbar, 1):
                lr_2 = lr_2.to(self.device, non_blocking=True)
                lr_3 = lr_3.to(self.device, non_blocking=True)
                lr_4 = lr_4.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                self.model.upscale_factor = 2
                output = tile_forward(self.model, 2, lr_2, tile_size=256, overlap=8)
                val_loss += self.criterion(output, target).item()
                self.metrics_ssim.update(output, target)

                self.model.upscale_factor = 3
                output = tile_forward(self.model, 3, lr_3, tile_size=256, overlap=8)
                val_loss += self.criterion(output, target).item()
                self.metrics_ssim.update(output, target)

                self.model.upscale_factor = 4
                output = tile_forward(self.model, 4, lr_4, tile_size=256, overlap=8)
                val_loss += self.criterion(output, target).item()
                self.metrics_ssim.update(output, target)

                pbar.set_postfix({"SSIM": f"{self.metrics_ssim.compute():.4f}"})

        self.history['validation'].append({
            'epoch': self.epoch,
            'loss': val_loss / len(self.val_loader) * 3,
            'ssim': self.metrics_ssim.compute().item()
        })

        return self.metrics_ssim.compute().item()

    def load_checkpoint(self):
        path = f"checkpoints/{self.config['model']['checkpoint_name']}_latest.pth"
        if not exists(path):
            print("No checkpoint to load")
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.best_val_ssim = checkpoint['best_val_ssim']
        self.history = checkpoint['history']
        print(f"Resumed from epoch {checkpoint['epoch']}")

    def save_checkpoint(self, type=""):
        path = f"checkpoints/{self.config['model']['checkpoint_name']}{type}.pth"
        torch.save({
            'epoch': self.epoch,
            'best_val_ssim': self.best_val_ssim,
            'history': self.history,
            'model_config': self.config['model'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Checkpoint saved to {path}")

    def train(self):
        print(f"Training model {self.config['model']['checkpoint_name']}")
        self.load_checkpoint()

        while self.epoch <= self.epochs:
            self.train_epoch()
            self.scheduler.step()

            if self.epoch % self.config['validate_every'] == 0 or self.epoch == self.epochs:
                val_ssim = self.validate()
                if val_ssim > self.best_val_ssim:
                    self.best_val_ssim = val_ssim
                    self.save_checkpoint()
                self.save_checkpoint('_latest')

            self.epoch += 1

        plot_training_history(self.history, self.config['model']['checkpoint_name'])
