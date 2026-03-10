from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import v2

from models import RegularModel
from utils.checkpoints import load_model_from_checkpoint
from utils.metrics import SSIM, PSNR


def get_unique_path(path):
    counter = 1
    temp_path = path
    while temp_path.exists():
        temp_path = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        counter += 1
    return temp_path


class ImageComparison:
    def __init__(self, lr_path, hr_path, device, upscale_factor):
        self.device = device
        self.upscale_factor = upscale_factor
        self.img_name = Path(lr_path).stem
        self.lr_image_fp = read_image(str(lr_path)).unsqueeze(0).float().div(255).to(self.device)
        self.hr_image = read_image(str(hr_path)).unsqueeze(0).float().to(self.device)
        self.ssim = SSIM(device=device, data_range=255.0)
        self.psnr = PSNR(device=device, data_range=255.0)

    def proccess_method(self, method):
        model = RegularModel(method, self.upscale_factor)
        model.eval()
        with torch.no_grad():
            output = model(self.lr_image_fp).clip(0, 1).mul(255).round()

        ssim = self.ssim(output, self.hr_image)
        psnr = self.psnr(output, self.hr_image)

        output = v2.functional.to_pil_image(output.squeeze().cpu().div(255))
        print('Processed:', method)
        return {
            'name': method, 'desc': f'{ssim:.4f}/{psnr:.2f}', 'img': output,
            'metrics': {'ssim': ssim, 'psnr': psnr}
        }

    def proccess_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, self.device)
        model.eval()
        with torch.no_grad():
            output = model(self.lr_image_fp).clip(0, 1).mul(255).round()

        ssim = self.ssim(output, self.hr_image)
        psnr = self.psnr(output, self.hr_image)

        output = v2.functional.to_pil_image(output.squeeze().cpu().div(255))
        print('Processed:', checkpoint_path.stem)
        return {
            'name': checkpoint['checkpoint_name'], 'desc': f'{ssim:.4f}/{psnr:.2f}', 'img': output,
            'metrics': {'ssim': ssim, 'psnr': psnr}
        }

    def compare(self, checkpoint_paths, methods, crop_box, save_dir='inference/comparison', dpi=150):
        """
        Args:
            crop_box: Tuple (x, y, width, height) - region za crop NA HR SLICI
        """
        # Process image with all models
        images = [self.proccess_checkpoint(ch) for ch in checkpoint_paths]
        images += [self.proccess_method(m) for m in methods]
        images.sort(key=lambda x: (x['metrics']['ssim'], x['metrics']['psnr']), reverse=True)
        # Add HR image
        hr_pil = v2.functional.to_pil_image(self.hr_image.squeeze().div(255).cpu())
        images.insert(0, {'name': 'Ground Truth', 'desc': 'SSIM/PSNR', 'img': hr_pil})
        # LR image for left side
        original_image = v2.functional.to_pil_image(self.lr_image_fp.squeeze().clamp(0, 1).cpu())

        grid_cols = min(6, len(images))
        grid_rows = int(np.ceil(len(images) / grid_cols))
        x, y, w, h = crop_box

        x_lr = x // self.upscale_factor
        y_lr = y // self.upscale_factor
        w_lr = w // self.upscale_factor
        h_lr = h // self.upscale_factor

        # Kreiraj figuru sa GridSpec
        fig = plt.figure(figsize=(4 * grid_cols, 3.5 * grid_rows))
        gs = fig.add_gridspec(grid_rows, grid_cols + 1, width_ratios=[2] + [1] * grid_cols, hspace=0.1, wspace=0.1)

        # Leva strana - LR slika sa pravougaonikom (na LR koordinatama)
        ax_original = fig.add_subplot(gs[:, 0])
        ax_original.imshow(original_image)
        ax_original.axis('off')
        ax_original.text(0.5, -0.05, self.img_name, transform=ax_original.transAxes, ha='center', va='top', fontsize=10)

        # Draw rectangle for crop patch on original image
        ax_original.add_patch(patches.Rectangle((x_lr, y_lr), w_lr, h_lr,
                                                linewidth=1, edgecolor='red', facecolor='none', linestyle='-'))

        # Comparison grid
        for i, method_img in enumerate(images):
            ax = fig.add_subplot(gs[i // grid_cols, i % grid_cols + 1])

            crop = np.array(method_img['img'])[y:y + h, x:x + w]
            ax.imshow(crop, interpolation='nearest')
            ax.axis('off')

            ax.text(0.5, -0.1, f'{method_img['name']}\n{method_img['desc']}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=10)

        save_path = Path(save_dir) / f'comparison_{self.img_name}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path = get_unique_path(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.2)
        print(f"Image saved to: {save_path}")
        plt.show()
