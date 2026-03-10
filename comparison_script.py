from pathlib import Path

import torch
from utils.comparator import ImageComparison

# Napravi grid uporedjivanja slike koji prikazuje delove slike povecane koristici razlicite metode i modele
# Implementacija u utils.comparator

if __name__ == "__main__":
    CROP_BOX = (1010, 730, 64, 64)  # (x, y, w, h)
    LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X2/0879x2.png'
    HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'

    # CROP_BOX = (744, 936, 96, 96)  # (x, y, w, h)
    # LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X2/0873x2.png'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0873.png'

    # CROP_BOX = (842, 685, 128, 128)  # (x, y, w, h)
    # LR_PATH = 'inference/comparison/input/F1.png'
    # HR_PATH = 'inference/comparison/input/F1_gt.png'
    #
    # CROP_BOX = (815, 388, 128, 128)  # (x, y, w, h)
    # LR_PATH = 'inference/comparison/input/R6.png'
    # HR_PATH = 'inference/comparison/input/R6_gt.png'

    # CROP_BOX = (1010, 730, 96, 96)  # (x, y, w, h)
    # LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X4/0879x4.png'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'

    UPSCALE_FACTOR = 2
    METHODS = ['nearest', 'bilinear', 'bicubic', 'lanczos']
    CHECKPOINT_PATHS = [
        Path("checkpoints/SR_EDSR_2x_32_256_r_best.pth"),
        Path("checkpoints/SR_EDSR_2x_16_64_best.pth"),
        Path("checkpoints/SR_EDSR_2x_2_48_best.pth"),
        Path("checkpoints/SR_RFDN_2x_4_256_best.pth"),
        Path("checkpoints/SR_RFDN_2x_4_128_best.pth"),
        Path("checkpoints/SR_RFDN_2x_1_128_best.pth"),
        Path("checkpoints/SR_RFDN_2x_4_52_best.pth"),
        Path("checkpoints/SR_RFDN_2x_2_48_best.pth"),
        Path("checkpoints/SR_IMDN_2x_4_52_best.pth"),
        Path("checkpoints/SR_CustomERN_2x_b2_w3_best.pth"),
        Path("checkpoints/SR_CustomERN_2x_b2_w2_best.pth"),
        Path("checkpoints/SR_CustomERN_2x_b2_w1_best.pth"),
        Path("checkpoints/SR_CustomERN_2x_b2_w0.5_best.pth"),
    ]
    # CHECKPOINT_PATHS = [
    #     Path("checkpoints/SR_EDSR_4x_6_12_best.pth"),
    #     Path("checkpoints/SR_RFDN_4x_6_48_best.pth"),
    #     Path("checkpoints/SR_RFDN_4x_1_128_best.pth"),
    #     Path("checkpoints/SR_RFDN_4x_2_128_best.pth"),
    #     Path("checkpoints/SR_RFDN_4x_6_52_best.pth"),
    #     Path("checkpoints/SR_IMDN_4x_6_64_best.pth"),
    #     Path("checkpoints/SR_CustomERN_4x_b2_w2_best.pth"),
    # ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparator = ImageComparison(lr_path=LR_PATH, hr_path=HR_PATH, device=device, upscale_factor=UPSCALE_FACTOR)
    comparator.compare(checkpoint_paths=CHECKPOINT_PATHS, methods=METHODS, crop_box=CROP_BOX, dpi=150)
