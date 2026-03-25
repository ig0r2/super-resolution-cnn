import copy
from pathlib import Path
from typing import Literal

import torch

from utils.checkpoints import load_model_from_checkpoint
from utils.csv_utils import save_to_csv
from utils.video.evaluator_perf_video import EvaluatorPerfVideo, Runtype
from utils.logger import Logger

# Evaluacija brzine modela za OpenCV video player demo

if __name__ == "__main__":
    UPSCALE_FACTOR: Literal[2, 3, 4] = 2
    RUNTYPE: Runtype = "tensorrt"

    CHECKPOINT_PATHS = [
        # Path("checkpoints/SR_EDSR_2x_1_32_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_32_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_4_32_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_6_12_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_48_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_52_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_4_52_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_16_64_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_96_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_1_128_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_128_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_1_256_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_2_256_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_4_256_best.pth"),
        # Path("checkpoints/SR_EDSR_2x_32_256_r_best.pth"),

        # Path("checkpoints/SR_RFDN_2x_1_4_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_48_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_48_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_52_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_48_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_52_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_16_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_16_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_6_48_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_6_52_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_64_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_64_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_6_64_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_96_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_96_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_96_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_128_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_128_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_128_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_1_256_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_2_256_best.pth"),
        # Path("checkpoints/SR_RFDN_2x_4_256_best.pth"),

        # Path("checkpoints/SR_IMDN_2x_1_48_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_2_48_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_4_16_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_5_16_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_4_52_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_1_64_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_6_64_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_1_96_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_2_96_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_1_128_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_2_128_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_1_256_best.pth"),
        # Path("checkpoints/SR_IMDN_2x_4_256_best.pth"),

        # Path("checkpoints/SR_CustomERN_2x_b2_w0.5_best.pth"),
        # Path("checkpoints/SR_CustomERN_2x_b2_w0.75_best.pth"),
        # Path("checkpoints/SR_CustomERN_2x_b1_w1_best.pth"),
        # Path("checkpoints/SR_CustomERN_2x_b2_w1_best.pth"),
        # Path("checkpoints/SR_CustomERN_2x_b3_w1_best.pth"),
        # Path("checkpoints/SR_CustomERN_2x_b4_w1_best.pth"),
        # Path("checkpoints/SR_CustomERN_2x_b2_w2_best.pth"),
        # Path("checkpoints/SR_CustomERN_2x_b3_w2_best.pth"),
        # Path("checkpoints/SR_CustomERN_2x_b2_w3_best.pth"),

        # Path("checkpoints/SR_EDSR_4x_6_12_best.pth"),
        # Path("checkpoints/SR_EDSR_4x_2_48_best.pth"),
        # Path("checkpoints/SR_EDSR_4x_4_48_best.pth"),
        # Path("checkpoints/SR_EDSR_4x_2_64_best.pth"),
        # Path("checkpoints/SR_RFDN_4x_2_48_best.pth"),
        # Path("checkpoints/SR_RFDN_4x_4_48_best.pth"),
        # Path("checkpoints/SR_RFDN_4x_6_48_best.pth"),
        # Path("checkpoints/SR_RFDN_4x_6_52_best.pth"),
        # Path("checkpoints/SR_RFDN_4x_1_128_best.pth"),
        # Path("checkpoints/SR_RFDN_4x_2_128_best.pth"),
        # Path("checkpoints/SR_IMDN_4x_2_48_best.pth"),
        # Path("checkpoints/SR_IMDN_4x_4_48_best.pth"),
        # Path("checkpoints/SR_IMDN_4x_6_64_best.pth"),
        # Path("checkpoints/SR_CustomERN_4x_b2_w2_best.pth"),
    ]

    #####################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    csv_path = f"results/results_{UPSCALE_FACTOR}x_FPS.csv"

    for checkpoint_path in CHECKPOINT_PATHS:
        checkpoint_path = Path(checkpoint_path)
        log_path = f"logs/evaluation/eval_{checkpoint_path.stem}.txt"

        with (Logger(log_path)):
            print("=" * 50)
            print(f"Checkpoint: {checkpoint_path.stem}")

            model, model_conf = load_model_from_checkpoint(checkpoint_path, device)
            name = model_conf['checkpoint_name']

            # Parameter number
            total_params = sum(p.numel() for p in model.parameters())
            print("-" * 30)
            print(f"Total parameters: {total_params:,}")
            print("-" * 30)

            # Performance
            perf_results = {'720p': "", '480p': "", '720p (128x128)': "", '480p (128x128)': "", '720p (256x256)': "",
                            '480p (256x256)': ""}
            perf_results['720p'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(720, 1280),
                                   runtype=RUNTYPE).evaluate())
            perf_results['480p'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(480, 854),
                                   runtype=RUNTYPE).evaluate())
            perf_results['720p (128x128)'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(720, 1280), tiled=True,
                                   tile_size=128,
                                   runtype=RUNTYPE).evaluate())
            perf_results['480p (128x128)'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(480, 854), tiled=True,
                                   tile_size=128,
                                   runtype=RUNTYPE, dont_export=True).evaluate())
            perf_results['720p (256x256)'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(720, 1280), tiled=True,
                                   tile_size=256,
                                   runtype=RUNTYPE).evaluate())
            perf_results['480p (256x256)'] = (
                EvaluatorPerfVideo(model=copy.deepcopy(model), name=name, image_size=(480, 854), tiled=True,
                                   tile_size=256,
                                   runtype=RUNTYPE, dont_export=True).evaluate())

            # Save to CSV
            save_to_csv({
                "model_name": checkpoint_path.stem,
                "params": total_params,
                "runtype": RUNTYPE,
                **perf_results,
            }, csv_path)
