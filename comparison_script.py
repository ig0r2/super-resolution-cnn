from pathlib import Path

import torch
from utils.comparator import ImageComparison

# Napravi grid uporedjivanja slike koji prikazuje delove slike povecane koristici razlicite metode i modele
# Implementacija u utils.comparator

if __name__ == "__main__":
    CROP_BOX = (505, 365, 32, 32)  # (x, y, w, h)
    LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X2/0879x2.png'
    HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'
    #
    CROP_BOX = (590, 290, 75, 75)  # (x, y, w, h)
    LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X2/0887x2.png'
    HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0887.png'

    # CROP_BOX = (372, 468, 48, 48)  # (x, y, w, h)
    # LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X2/0873x2.png'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0873.png'

    # CROP_BOX = (421, 342, 64, 64)  # (x, y, w, h)
    # LR_PATH = 'inference/comparison/input/F1.png'
    # HR_PATH = 'inference/comparison/input/F1_gt.png'
    #
    # CROP_BOX = (407, 194, 64, 64)  # (x, y, w, h)
    # LR_PATH = 'inference/comparison/input/R6.png'
    # HR_PATH = 'inference/comparison/input/R6_gt.png'

    # CROP_BOX = (863, 319, 128, 128)  # (x, y, w, h)
    # LR_PATH = 'inference/comparison/input/wrc.jpg'
    # HR_PATH = 'inference/comparison/input/wrc_gt.jpg'
    #
    # CROP_BOX = (505, 365, 32, 32)  # (x, y, w, h)
    # LR_PATH = 'inference/comparison/input/0879x2_45.jpg'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'
    #
    # CROP_BOX = (505, 365, 32, 32)  # (x, y, w, h)
    # LR_PATH = 'inference/comparison/input/0879x2_85.jpg'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'

    # CROP_BOX = (110, 628, 96, 96)  # (x, y, w, h)
    # LR_PATH = 'inference/comparison/input/0879x2_45.jpg'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'

    UPSCALE_FACTOR = 2
    METHODS = ['nearest', 'bilinear', 'bicubic', 'lanczos']
    CHECKPOINT_PATHS = [
        # Path("checkpoints/SR_EDSR_2x_32_256_r.pth"),
        Path("checkpoints/SR_EDSR_2_52.pth"),
        Path("checkpoints/SR_RFDN_4_256.pth"),
        # Path("checkpoints/SR_RFDN_4_128.pth"),
        Path("checkpoints/SR_RFDN_1_128.pth"),
        Path("checkpoints/SR_RFDN_2_48.pth"),
        Path("checkpoints/SR_IMDN_2_48.pth"),
        Path("checkpoints/SR_FastEDSR_4_64.pth"),
        Path("checkpoints/SR_FastEDSR_4_128.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_2_64.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_32.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_48.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_64.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_128.pth"),
        Path("checkpoints/SR_FastEDSR_jpeg_4_256.pth"),
    ]

    ############### 3 #############################
    # UPSCALE_FACTOR = 3
    #
    # CROP_BOX = (336, 243, 32, 32)  # (x, y, w, h)
    # LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X3/0879x3.png'
    # HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'

    ############### 4 #############################
    UPSCALE_FACTOR = 4

    CROP_BOX = (252, 182, 48, 48)  # (x, y, w, h)
    LR_PATH = 'data/DIV2K/DIV2K_valid_LR_bicubic/X4/0879x4.png'
    HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'

    CROP_BOX = (252, 182, 48, 48)  # (x, y, w, h)
    LR_PATH = 'inference/comparison/input/0879x4_85.jpg'
    HR_PATH = 'data/DIV2K/DIV2K_valid_HR/0879.png'

    ##################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparator = ImageComparison(lr_path=LR_PATH, hr_path=HR_PATH, device=device, upscale_factor=UPSCALE_FACTOR)
    comparator.compare(checkpoint_paths=CHECKPOINT_PATHS, methods=METHODS, crop_box=CROP_BOX, dpi=150)
