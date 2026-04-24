import torch

from torchvision.io import encode_jpeg, decode_jpeg
import torchvision.transforms.v2.functional as TF


def ensure_rgb(img: torch.Tensor) -> torch.Tensor:
    return TF.grayscale_to_rgb(img) if img.shape[0] != 3 else img


def crop_to_match(img, target_h, target_w):
    diff_h = img.shape[1] - target_h
    diff_w = img.shape[2] - target_w
    if diff_h == 0 and diff_w == 0:
        return img
    return TF.crop_image(img, diff_h // 2, diff_w // 2, target_h, target_w)


def sharpen_image(tensor: torch.Tensor, kernel_size=5, sigma=1.0, strength=0.5):
    """[B, C, H, W] or [C, H, W], float -> float"""
    single_img = tensor.ndim == 3
    if single_img:
        tensor = tensor.unsqueeze(0)  # -> [1, C, H, W]

    blurred = TF.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    sharpened = tensor + strength * (tensor - blurred)
    sharpened = sharpened.clamp_(0.0, 1.0)

    return sharpened.squeeze(0) if single_img else sharpened


def apply_jpeg_compression(tensor: torch.Tensor, quality: int):
    """[B, C, H, W] or [C, H, W], unit8 -> unit8"""
    single_img = tensor.ndim == 3
    if single_img:
        tensor = tensor.unsqueeze(0)

    compressed = torch.stack([decode_jpeg(encode_jpeg(img, quality=quality)) for img in tensor])
    return compressed.squeeze(0) if single_img else compressed


def jpeg_quality_for_index(index: int, n_images: int = 100, min_q: int = 20, max_q: int = 100) -> int:
    """Map a dataset index to a deterministic JPEG quality level.

    Spreads quality values evenly across [min_q, max_q] so that a fixed-size
    test set has consistent, reproducible degradation per image.
    """
    return (index + 1) * (max_q - min_q) // n_images + min_q
