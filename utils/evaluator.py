import sys
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.metrics import PSNR, SSIM, LPIPS


class Evaluator:
    """
    Evaluira model sa test_set
    Podrzava testiranje modela sa half precision
    Metrike se racunaju tako sto se prvo pretvori output modela u pixel reprezentaciju opseg [0-255]
    """

    def __init__(self, test_set, device, model=None, use_half=False, upscale_factor=2):
        self.model = model
        self.device = device
        self.use_half = use_half
        self.upscale_factor = upscale_factor

        self.test_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=1, persistent_workers=True,
                                      pin_memory=True)

        self.psnr = PSNR(device=device, data_range=255.0)
        self.ssim = SSIM(device=device, data_range=255.0)
        self.lpips = LPIPS(device=device)
        self.loss = nn.L1Loss()

    def set_model(self, model, use_half=None):
        self.model = model
        if use_half is not None: self.use_half = use_half

        if self.use_half:
            self.model = self.model.half()
        return self

    def evaluate(self):
        total_loss = 0
        self.psnr.reset()
        self.ssim.reset()
        self.lpips.reset()

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f'Evaluation', file=sys.stdout)
            for _, (input, target) in enumerate(pbar):
                # input, target - uint8[0-255]
                input, target = input.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                # input_fp, target_fp - normalize to [0-1]
                if self.use_half:
                    input_fp, target_fp = input.half().div(255.0), target.half().div(255.0)
                else:
                    input_fp, target_fp = input.float().div(255.0), target.float().div(255.0)

                output_fp = self.model(input_fp)
                output_fp = output_fp.clip(0, 1)
                output_int = output_fp.mul(255).round().float()

                total_loss += self.loss(output_fp, target_fp).item()
                self.lpips.update(output_fp, target_fp)
                self.psnr.update(output_int, target)
                self.ssim.update(output_int, target)

        n = len(self.test_loader)
        print(f'Loss: {total_loss / n:.4f}')
        print(f'PSNR: {self.psnr.compute():.4f}')
        print(f'SSIM: {self.ssim.compute():.4f}')
        print(f'LPIPS: {self.lpips.compute():.4f}')

        return {
            'LPIPS': f"{self.lpips.compute():.4f}",
            'SSIM': f"{self.ssim.compute():.4f}",
            'PSNR': f"{self.psnr.compute():.4f}",
            'Loss': f"{total_loss / n:.4f}",
        }
