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
    #
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

    UPSCALE_FACTOR = 2
    METHODS = ['nearest', 'bilinear', 'bicubic', 'lanczos']
    CHECKPOINT_PATHS = [
        Path("checkpoints/SR_EDSR_2x_32_256_r.pth"),
        Path("checkpoints/SR_EDSR_2x_16_64.pth"),
        Path("checkpoints/SR_EDSR_2x_2_48.pth"),
        Path("checkpoints/SR_RFDN_2x_4_256.pth"),
        Path("checkpoints/SR_RFDN_2x_4_128.pth"),
        Path("checkpoints/SR_RFDN_2x_1_128.pth"),
        Path("checkpoints/SR_RFDN_2_48.pth"),
        Path("checkpoints/SR_IMDN_2x_4_52.pth"),
        Path("checkpoints/SR_FastEDSR_2_8.pth"),
        Path("checkpoints/SR_FastEDSR_2_16.pth"),
        Path("checkpoints/SR_FastEDSR_2_64.pth"),
        Path("checkpoints/SR_FastEDSR_4_64.pth"),
        Path("checkpoints/SR_FastEDSR_4_128.pth"),
    ]

    # UPSCALE_FACTOR = 3
    # CROP_BOX = (1000, 720, 192, 192)  # (x, y, w, h)
    # LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X3/0879x3.png'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'
    #
    # CHECKPOINT_PATHS = [
    #     Path("checkpoints/SR_EDSR_3x_6_12.pth"),
    #     Path("checkpoints/SR_EDSR_3x_4_48.pth"),
    #     Path("checkpoints/SR_RFDN_3x_2_48.pth"),
    #     Path("checkpoints/SR_RFDN_3x_1_128.pth"),
    #     Path("checkpoints/SR_RFDN_3x_2_128.pth"),
    #     Path("checkpoints/SR_RFDN_3x_6_48.pth"),
    #     Path("checkpoints/SR_IMDN_3x_6_64.pth"),
    #     Path("checkpoints/SR_FastEDSR_3x_2_8.pth"),
    #     Path("checkpoints/SR_FastEDSR_3x_2_16.pth"),
    #     Path("checkpoints/SR_FastEDSR_3x_2_64.pth"),
    #     Path("checkpoints/SR_FastEDSR_3x_4_16.pth"),
    #     Path("checkpoints/SR_FastEDSR_3x_4_64.pth"),
    #     Path("checkpoints/SR_FastEDSR_3x_4_128.pth"),
    # ]

    # UPSCALE_FACTOR = 4
    # CROP_BOX = (1000, 720, 192, 192)  # (x, y, w, h)
    # LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X4/0879x4.png'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'

    # CROP_BOX = (744, 936, 192, 192)  # (x, y, w, h)
    # LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X4/0873x4.png'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0873.png'

    # CHECKPOINT_PATHS = [
    #     Path("checkpoints/SR_EDSR_4x_6_12.pth"),
    #     Path("checkpoints/SR_EDSR_4x_4_48.pth"),
    #     Path("checkpoints/SR_RFDN_4x_2_48.pth"),
    #     Path("checkpoints/SR_RFDN_4x_1_128.pth"),
    #     Path("checkpoints/SR_RFDN_4x_2_128.pth"),
    #     Path("checkpoints/SR_RFDN_4x_6_48.pth"),
    #     Path("checkpoints/SR_IMDN_4x_6_64.pth"),
    #     Path("checkpoints/SR_FastEDSR_4x_2_8.pth"),
    #     Path("checkpoints/SR_FastEDSR_4x_2_16.pth"),
    #     Path("checkpoints/SR_FastEDSR_4x_2_64.pth"),
    #     Path("checkpoints/SR_FastEDSR_4x_4_16.pth"),
    #     Path("checkpoints/SR_FastEDSR_4x_4_64.pth"),
    #     Path("checkpoints/SR_FastEDSR_4x_4_128.pth"),
    # ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparator = ImageComparison(lr_path=LR_PATH, hr_path=HR_PATH, device=device, upscale_factor=UPSCALE_FACTOR)
    comparator.compare(checkpoint_paths=CHECKPOINT_PATHS, methods=METHODS, crop_box=CROP_BOX, dpi=150)
