from pathlib import Path

import torch
from utils.comparator import ImageComparison

# Napravi grid uporedjivanja slike koji prikazuje delove slike povecane koristici razlicite metode i modele
# Implementacija u utils.comparator

if __name__ == "__main__":
    CROP_BOX = (1010, 730, 64, 64)  # (x, y, w, h)
    LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X2/0879x2.png'
    HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'
    #
    # CROP_BOX = (1180, 580, 150, 150)  # (x, y, w, h)
    # LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X2/0887x2.png'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0887.png'

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
        Path("checkpoints/SR_EDSR_2x_32_256_r.pth"),
        Path("checkpoints/SR_EDSR_2x_16_64.pth"),
        Path("checkpoints/SR_EDSR_2x_2_48.pth"),
        Path("checkpoints/SR_RFDN_2x_4_256.pth"),
        Path("checkpoints/SR_RFDN_2x_4_128.pth"),
        Path("checkpoints/SR_RFDN_2x_1_128.pth"),
        Path("checkpoints/SR_RFDN_2x_4_52.pth"),
        Path("checkpoints/SR_RFDN_2x_2_48.pth"),
        Path("checkpoints/SR_IMDN_2x_4_52.pth"),
        Path("checkpoints/SR_CustomERN_2x_b2_w3.pth"),
        Path("checkpoints/SR_CustomERN_2x_b2_w2.pth"),
        Path("checkpoints/SR_CustomERN_2x_b2_w1.pth"),
        Path("checkpoints/SR_CustomERN_2x_b2_w0.5.pth"),
    ]
    # CHECKPOINT_PATHS = [
    #     Path("checkpoints/SR_EDSR_4x_6_12.pth"),
    #     Path("checkpoints/SR_RFDN_4x_6_48.pth"),
    #     Path("checkpoints/SR_RFDN_4x_1_128.pth"),
    #     Path("checkpoints/SR_RFDN_4x_2_128.pth"),
    #     Path("checkpoints/SR_RFDN_4x_6_52.pth"),
    #     Path("checkpoints/SR_IMDN_4x_6_64.pth"),
    #     Path("checkpoints/SR_CustomERN_4x_b2_w2.pth"),
    # ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparator = ImageComparison(lr_path=LR_PATH, hr_path=HR_PATH, device=device, upscale_factor=UPSCALE_FACTOR)
    comparator.compare(checkpoint_paths=CHECKPOINT_PATHS, methods=METHODS, crop_box=CROP_BOX, dpi=150)
